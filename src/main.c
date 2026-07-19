#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "image_io.h"
#include "quadtree.h"
#include "color.h"
#include "sat.h"
#include "svg_writer.h"
#include "svbc_writer.h"
#include "optimizer.h"
#include "contour.h"
#include "codebook_db.h"
#include "nvdr_types.h"

// PaletteEntry, IntList, NodeGradient definidos em nvdr_types.h

static int find_or_add_palette(PaletteEntry* cb, int* cb_size, QuadNode* node, int has_grad, int grad_id) {
    for (int i = 0; i < *cb_size; i++) {
        if (cb[i].has_gradient == has_grad) {
            if (has_grad) {
                if (cb[i].gradient_id == grad_id) { cb[i].freq++; return i; }
            } else {
                if (cb[i].r == node->avg_r && cb[i].g == node->avg_g && cb[i].b == node->avg_b) {
                    cb[i].freq++; return i;
                }
            }
        }
    }
    int idx = (*cb_size)++;
    cb[idx].r = node->avg_r; cb[idx].g = node->avg_g; cb[idx].b = node->avg_b;
    cb[idx].has_gradient = has_grad; cb[idx].gradient_id = grad_id; cb[idx].freq = 1;
    if (has_grad) snprintf(cb[idx].hex_fill, sizeof(cb[idx].hex_fill), "url(#g%d)", grad_id);
    else color_to_hex(node->avg_r, node->avg_g, node->avg_b, cb[idx].hex_fill);
    return idx;
}

// Write CSS Indexed Palette (Paleta Indexada — zero-loss compression)
static void write_css_palette(SVGWriter* svg, PaletteEntry* palette, int palette_size) {
    fprintf(svg->file, "<style>");
    for (int p = 0; p < palette_size; p++) {
        fprintf(svg->file, ".c%d{fill:%s}", p, palette[p].hex_fill);
    }
    fprintf(svg->file, "</style>\n");
}

// Comparator for qsort: QuadNode* by (y, x)
static int cmp_node_ptr_yx(const void* a, const void* b) {
    const QuadNode* na = *(const QuadNode* const*)a;
    const QuadNode* nb = *(const QuadNode* const*)b;
    if (na->y != nb->y) return na->y - nb->y;
    return na->x - nb->x;
}

// Write a PRS layer using CSS classes + greedy horizontal merge + relative coords
static void write_layer(SVGWriter* svg, IntList* layer, QuadTree* qt,
                        PaletteEntry* palette, int palette_size, int* node_to_palette) {
    (void)palette_size;
    for (int p = 0; p < palette_size; p++) {
        int count_in_layer = 0;
        for (int i = 0; i < layer->count; i++) {
            if (node_to_palette[layer->node_indices[i]] == p) count_in_layer++;
        }
        if (count_in_layer == 0) continue;

        if (palette[p].has_gradient) {
            for (int i = 0; i < layer->count; i++) {
                int ni = layer->node_indices[i];
                if (node_to_palette[ni] == p) {
                    QuadNode* node = &qt->nodes[ni];
                    fprintf(svg->file, "<rect class=\"c%d\" x=\"%d\" y=\"%d\" width=\"%d\" height=\"%d\"/>\n",
                            p, node->x, node->y, node->w, node->h);
                }
            }
        } else {
            QuadNode** merged = malloc(layer->count * sizeof(QuadNode*));
            if (!merged) continue;
            int mc = 0;
            for (int i = 0; i < layer->count; i++) {
                int ni = layer->node_indices[i];
                if (node_to_palette[ni] == p) merged[mc++] = &qt->nodes[ni];
            }
            if (mc == 0) {
                free(merged);
                continue;
            }
            qsort(merged, mc, sizeof(QuadNode*), cmp_node_ptr_yx);
            fprintf(svg->file, "<path class=\"c%d\" d=\"", p);
            int cx = merged[0]->x, cy = merged[0]->y, cw = merged[0]->w, ch = merged[0]->h;
            int pen_x = 0, pen_y = 0;
            for (int i = 1; i < mc; i++) {
                QuadNode* n = merged[i];
                if (n->y == cy && n->h == ch && cx + cw == n->x) {
                    cw += n->w;
                } else {
                    int dx = cx - pen_x, dy = cy - pen_y;
                    fprintf(svg->file, "m%d %dh%dv%dh-%dz", dx, dy, cw, ch, cw);
                    pen_x = cx; pen_y = cy;
                    cx = n->x; cy = n->y; cw = n->w; ch = n->h;
                }
            }
            int dx = cx - pen_x, dy = cy - pen_y;
            fprintf(svg->file, "m%d %dh%dv%dh-%dz", dx, dy, cw, ch, cw);
            fprintf(svg->file, "\"/>\n");
            free(merged);
        }
    }
}

static long get_file_size(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return 0;
    fseek(f, 0, SEEK_END);
    long s = ftell(f);
    fclose(f);
    return s;
}

int main(int argc, char** argv) {
    const char* input_path = NULL;
    const char* output_path = NULL;
    const char* codebook_db_path = NULL;   /* optional persisted codebook */
    int min_tile_size = -1;
    int min_tile_size_set = 0;   // B3: distinguish "not passed" from "passed 0"
    float homo_thresh = -1.0f;
    int homo_thresh_set = 0;     // B3: distinguish "not passed" from "user explicitly chose 0"
    int export_svg = 1;
    int export_svbc = 1;
    int logo_mode = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--format") == 0 && i + 1 < argc) {
            if (strcmp(argv[i+1], "svg") == 0) { export_svg = 1; export_svbc = 0; }
            else if (strcmp(argv[i+1], "svbc") == 0) { export_svg = 0; export_svbc = 1; }
            else if (strcmp(argv[i+1], "all") == 0) { export_svg = 1; export_svbc = 1; }
            i++;
        } else if (strcmp(argv[i], "--logo") == 0) {
            logo_mode = 1;
        } else if (strcmp(argv[i], "--codebook-db") == 0 && i + 1 < argc) {
            codebook_db_path = argv[++i];
        } else if (!input_path) {
            input_path = argv[i];
        } else if (!output_path) {
            output_path = argv[i];
        } else if (!min_tile_size_set) {
            min_tile_size = atoi(argv[i]);
            min_tile_size_set = 1;
        } else if (!homo_thresh_set) {
            homo_thresh = (float)atof(argv[i]);
            homo_thresh_set = 1;
        }
    }

    if (!input_path || !output_path) {
        fprintf(stderr, "Usage: image_to_svg <input> <output_base> [min_tile] [homo_thresh] [--format svg|svbc|all]\n");
        return 1;
    }

    // Configuração NVDR Otimizada: balanço fidelidade vs. compactação
    // homo_threshold=0.01 preserva bordas de formas/logos/texto
    SVGConfig cfg = {
    .min_tile_size = 2,       // Snap-to-Grid: múltiplo de 2, evita 1px noise
    .max_depth = 10,          // Suficiente para detalhes finos a 1024px
    .homo_threshold = 0.01f,  // Preserva bordas nítidas (logos, texto, ícones)
    .semantic_layers = 1,
    .psq_sky_mult = 4.0f,
    .psq_midground_mult = 2.0f
    };
    
    if (min_tile_size_set && min_tile_size > 0) cfg.min_tile_size = min_tile_size;
    if (homo_thresh_set && homo_thresh >= 0.0f) cfg.homo_threshold = homo_thresh;

    // Logo mode: pixel-perfect settings for small mono/unicolor images
    if (logo_mode) {
        cfg.min_tile_size = 1;          // pixel-perfect edges
        cfg.homo_threshold = 0.005f;    // tight — sharp boundaries
        cfg.psq_sky_mult = 1.0f;        // no sky/midground layering
        cfg.psq_midground_mult = 1.0f;  // all layers equal
        cfg.max_depth = 12;             // deeper recursion for small icons
        // Allow user overrides even in logo mode
        if (min_tile_size_set && min_tile_size > 0) cfg.min_tile_size = min_tile_size;
        if (homo_thresh_set && homo_thresh >= 0.0f) cfg.homo_threshold = homo_thresh;
    }

    // Quality profile: quando o threshold é muito baixo, evitar PSQ agressivo.
    if (!logo_mode && cfg.homo_threshold <= 0.0015f) {
        cfg.psq_sky_mult = 1.0f;
        cfg.psq_midground_mult = 1.0f;
    }


#ifdef _OPENMP
    double t_start = omp_get_wtime();
#else
    clock_t t_start = clock();
#endif

    Image img;
    if (image_load(&img, input_path) != 0) {
        fprintf(stderr, "error: failed to load image '%s'\n", input_path);
        return 1;
    }

    // === Build SAT (Summed Area Table) for O(1) color queries ===
    SAT sat;
    if (sat_build(&sat, &img) != 0) {
        fprintf(stderr, "error: failed to allocate SAT for %dx%d image\n", img.width, img.height);
        image_free(&img);
        return 1;
    }

    // === ADAPTIVE COMPLEXITY DETECTION (v0.8 §1.3: perceptual homogeneity) ===
    // Quick pre-scan: sample ~2% of pixels to measure average gradient
    // High gradient = organic photo (loosen thresholds for compression)
    // Low gradient = icon/logo/UI (keep tight thresholds for sharpness)
    float img_complexity = 0.0f;
    {
        long total_diff = 0;
        int samples = 0;
        int step = 4; // sample every 4th pixel in each direction
        for (int py = 0; py < img.height - 1; py += step) {
            for (int px = 0; px < img.width - 1; px += step) {
                unsigned char* p0 = image_pixel(&img, px, py);
                unsigned char* p1 = image_pixel(&img, px + 1, py);
                unsigned char* p2 = image_pixel(&img, px, py + 1);
                // Horizontal + vertical gradient magnitude
                total_diff += abs((int)p0[0] - (int)p1[0]) + abs((int)p0[1] - (int)p1[1]) + abs((int)p0[2] - (int)p1[2]);
                total_diff += abs((int)p0[0] - (int)p2[0]) + abs((int)p0[1] - (int)p2[1]) + abs((int)p0[2] - (int)p2[2]);
                samples++;
            }
        }
        img_complexity = samples > 0 ? (float)total_diff / (float)samples : 0.0f;
    }

    // Adaptive thresholds based on complexity — quality-first
    float cull_thresh = 14.0f;
    int quant_step = 4;
    int coalesce_color_thresh = 8;

    if (logo_mode) {
        // Logo mode: minimal processing, exact colors
        cull_thresh = 0.0f;             // no importance culling
        quant_step = 0;                 // no quantization — keep exact colors
        coalesce_color_thresh = 2;      // merge only near-identical colors
    } else if (img_complexity > 30.0f) {
        // High complexity: organic photo — gentle adjustments only
        // B3: only override if user did NOT explicitly pass a threshold.
        if (!homo_thresh_set || homo_thresh < 0.0f) cfg.homo_threshold = 0.015f;
        cull_thresh = 16.0f;
        quant_step = 4;
        coalesce_color_thresh = 10;
    } else if (img_complexity > 15.0f) {
        // Medium complexity
        if (!homo_thresh_set || homo_thresh < 0.0f) cfg.homo_threshold = 0.012f;
        cull_thresh = 15.0f;
        quant_step = 4;
        coalesce_color_thresh = 10;
    }
    // Low complexity (<15): keep defaults — logo/icon/UI mode

    int estimated_nodes = (img.width / cfg.min_tile_size + 1) *
                          (img.height / cfg.min_tile_size + 1) * 2;
    if (estimated_nodes < 10000) estimated_nodes = 10000;

    // Em qualidade máxima, algumas imagens excedem a estimativa inicial.
    // Fazemos retry com capacidade crescente para evitar falha intermitente.
    const int max_nodes_cap = 4000000;
    int max_nodes = estimated_nodes;
    if (max_nodes > max_nodes_cap) max_nodes = max_nodes_cap;

    QuadTree qt;
    int build_ok = 0;
    while (!build_ok) {
        if (quadtree_init(&qt, max_nodes) != 0) {
            fprintf(stderr, "error: failed to allocate quadtree pool (%d nodes)\n", max_nodes);
            image_free(&img);
            return 1;
        }
        if (quadtree_build(&qt, &img, &sat, &cfg) >= 0) {
            build_ok = 1;
        } else {
            quadtree_free(&qt);
            if (max_nodes >= max_nodes_cap) {
                fprintf(stderr, "error: quadtree build exceeded max pool (%d nodes)\n", max_nodes_cap);
                image_free(&img);
                return 1;
            }
            int grown = max_nodes * 2;
            max_nodes = (grown > max_nodes_cap) ? max_nodes_cap : grown;
        }
    }

    // SAT no longer needed after quadtree build
    sat_free(&sat);

    // === Optional persisted codebook (cross-file dedup) ===
    // If user passed --codebook-db PATH, load any prior persisted
    // palette index. This is the bootstrap of Fluid Mechanism A from
    // the v0.14 spec — colors seen in earlier runs get a weight bonus
    // when competing for iLUT slots in this one.
    CodebookDB cb_db;
    codebook_db_init(&cb_db);
    int codebook_loaded = 0;
    if (codebook_db_path) {
        if (codebook_db_load(&cb_db, codebook_db_path) == 0) {
            codebook_loaded = 1;
        } else {
            fprintf(stderr, "warning: codebook-db at '%s' unreadable; starting fresh\n",
                    codebook_db_path);
            codebook_db_free(&cb_db);
            codebook_db_init(&cb_db);
        }
    }

#ifdef _OPENMP
    double t_built = omp_get_wtime();
#endif

    int leaves_before = optimizer_count_leaves(&qt, 0);
    int is_ultra = (cfg.homo_threshold < 0.005f);
    int is_max_quality = (cfg.min_tile_size <= 1 && cfg.homo_threshold <= 0.0015f);

    int ilut_size = logo_mode ? 64 : (is_ultra ? 1024 : (img_complexity > 30.0f ? 512 : 256));

    // Em qualidade máxima, prioriza continuidade tonal:
    // desliga culling/fusões agressivas que causam "saltos" no céu.
    if (is_max_quality) {
        cull_thresh = 0.0f;
        coalesce_color_thresh = 0;
    } else if (!is_ultra && (img.width <= 1280 || img.height <= 720)) {
        // Em resoluções menores, reduzir quantização para evitar posterização.
        quant_step = 2;
        if (coalesce_color_thresh > 4) coalesce_color_thresh = 4;
    }

    optimizer_apply_importance_cull(&qt, cull_thresh);
    if (!is_ultra && quant_step > 0) {
        optimizer_quantize(&qt, quant_step);
    }
    if (!is_ultra) {
        /* MVP: passed codebook only affects what we record, not what
         * iLUT picks. Future work (Mechanism A per spec §4.3) can
         * seed the ColorWeight array here directly. */
        optimizer_apply_ilut(&qt, ilut_size);
    }

    /* Record observed colors into the codebook so the next run carries
     * what we learned this time. We do this AFTER iLUT (so the colors
     * we record are the ones the iLUT actually picked) but BEFORE
     * coalesce (so coalesce can still apply to a fresh tree). */
    if (codebook_loaded || codebook_db_path) {
        for (int i = 0; i < qt.count; i++) {
            if (!qt.nodes[i].is_leaf) continue;
            QuadNode* n = &qt.nodes[i];
            int area = n->w * n->h;
            /* area can be 0; treat as 1 so we don't drop a tiny color */
            if (area < 1) area = 1;
            codebook_db_observe(&cb_db, n->avg_r, n->avg_g, n->avg_b,
                                area, /*run_id=*/1);
        }
    }

    optimizer_coalesce(&qt, 0, coalesce_color_thresh);
    optimizer_rect_coalesce(&qt, coalesce_color_thresh);
    // Skip blend detection in logo mode — logos want hard, clean edges
    if (!logo_mode) {
        optimizer_detect_blends(&qt);
    }

#ifdef _OPENMP
    double t_opt = omp_get_wtime();
#endif

    int leaves_after = optimizer_count_leaves(&qt, 0);

    // PRS Classification (v0.14.1 Section 1.2)
    // B4: each IntList's backing array is malloc'd and may return NULL
    // (especially for empty quadtrees where qt.count is 0 — that is
    // implementation-defined). We bail out cleanly if any allocation
    // fails.
    if (qt.count <= 0) {
        fprintf(stderr, "error: empty quadtree (%d nodes)\n", qt.count);
        quadtree_free(&qt); image_free(&img);
        return 1;
    }
    IntList prs_anchor = {NULL, 0};
    IntList prs_r1     = {NULL, 0};
    IntList prs_r2     = {NULL, 0};
    prs_anchor.node_indices = (int*)malloc((size_t)qt.count * sizeof(int));
    prs_r1.node_indices     = (int*)malloc((size_t)qt.count * sizeof(int));
    prs_r2.node_indices     = (int*)malloc((size_t)qt.count * sizeof(int));
    if (!prs_anchor.node_indices || !prs_r1.node_indices || !prs_r2.node_indices) {
        fprintf(stderr, "error: failed to allocate PRS layer buffers (%d nodes)\n", qt.count);
        free(prs_anchor.node_indices);
        free(prs_r1.node_indices);
        free(prs_r2.node_indices);
        quadtree_free(&qt); image_free(&img);
        return 1;
    }
    
    // Quality detection for Ultra mode

    for (int i = 0; i < qt.count; i++) {
        QuadNode* node = &qt.nodes[i];
        if (!node->is_leaf) continue;
        int max_dim = node->w > node->h ? node->w : node->h;
        if (max_dim >= 16)      prs_anchor.node_indices[prs_anchor.count++] = i;
        else if (max_dim >= 4)  prs_r1.node_indices[prs_r1.count++] = i;
        else                    prs_r2.node_indices[prs_r2.count++] = i;
    }

    // Gradient detection (blocks >= 8px) — both vertical AND horizontal
    NodeGradient* grad_info = (NodeGradient*)calloc((size_t)qt.count, sizeof(NodeGradient));
    int total_gradients = 0;
    if (grad_info) {
        for (int i = 0; i < qt.count; i++) {
            QuadNode* node = &qt.nodes[i];
            if (!node->is_leaf || node->w < 8 || node->h < 8) continue;
            unsigned char tr, tg, tb, br, bg, bb;
            // Try vertical gradient first
            if (color_gradient_v(&img, node->x, node->y, node->w, node->h,
                                 &tr, &tg, &tb, &br, &bg, &bb, 10)) {
                grad_info[i].has_gradient = 1;
                grad_info[i].grad_dir = 0; // vertical
                grad_info[i].top_r = tr; grad_info[i].top_g = tg; grad_info[i].top_b = tb;
                grad_info[i].bot_r = br; grad_info[i].bot_g = bg; grad_info[i].bot_b = bb;
                grad_info[i].gradient_id = total_gradients++;
            }
            // Try horizontal gradient if no vertical found
            else if (color_gradient_h(&img, node->x, node->y, node->w, node->h,
                                      &tr, &tg, &tb, &br, &bg, &bb, 10)) {
                grad_info[i].has_gradient = 1;
                grad_info[i].grad_dir = 1; // horizontal
                grad_info[i].top_r = tr; grad_info[i].top_g = tg; grad_info[i].top_b = tb;
                grad_info[i].bot_r = br; grad_info[i].bot_g = bg; grad_info[i].bot_b = bb;
                grad_info[i].gradient_id = total_gradients++;
            }
        }
    }

    // GCT Unified Palette (Codebook Universal per-image)
    PaletteEntry* palette = (PaletteEntry*)calloc((size_t)qt.count, sizeof(PaletteEntry));
    int* node_to_palette = (int*)calloc((size_t)qt.count, sizeof(int));
    int palette_size = 0;

    IntList* prs_layers[] = {&prs_anchor, &prs_r1, &prs_r2};
    for (int l = 0; l < 3; l++) {
        IntList* layer = prs_layers[l];
        for (int i = 0; i < layer->count; i++) {
            int ni = layer->node_indices[i];
            QuadNode* node = &qt.nodes[ni];
            int has_grad = grad_info && grad_info[ni].has_gradient;
            int grad_id = has_grad ? grad_info[ni].gradient_id : 0;
            node_to_palette[ni] = find_or_add_palette(palette, &palette_size, node, has_grad, grad_id);
        }
    }

    // Background color from root
    char bg_hex[8];
    color_to_hex(qt.nodes[0].avg_r, qt.nodes[0].avg_g, qt.nodes[0].avg_b, bg_hex);

    // === Export SVG directly to output_path ===
    if (export_svg) {
        SVGWriter svg_final;
        if (svg_open(&svg_final, output_path, img.width, img.height) == 0) {
            svg_begin_defs(&svg_final);
            if (grad_info) {
                for (int i = 0; i < qt.count; i++) {
                    if (!grad_info[i].has_gradient) continue;
                    char th[8], bh[8];
                    color_to_hex(grad_info[i].top_r, grad_info[i].top_g, grad_info[i].top_b, th);
                    color_to_hex(grad_info[i].bot_r, grad_info[i].bot_g, grad_info[i].bot_b, bh);
                    if (grad_info[i].grad_dir == 0) {
                        svg_linear_gradient_v(&svg_final, grad_info[i].gradient_id, th, bh);
                    } else {
                        fprintf(svg_final.file,
                            "<linearGradient id=\"g%d\" x1=\"0\" y1=\"0\" x2=\"1\" y2=\"0\">"
                            "<stop offset=\"0%%\" stop-color=\"%s\"/>"
                            "<stop offset=\"100%%\" stop-color=\"%s\"/>"
                            "</linearGradient>\n",
                            grad_info[i].gradient_id, th, bh);
                    }
                }
            }
            svg_end_defs(&svg_final);
            write_css_palette(&svg_final, palette, palette_size);
            svg_rect(&svg_final, 0, 0, img.width, img.height, bg_hex);

            fprintf(svg_final.file, "<g id=\"NVDR_Anchor\">\n");
            /* Anchor layer: use smooth contour paths for curves/circles.
             * Falls back to rect-merged paths for small/rectangular regions. */
            contour_write_layer_smooth(svg_final.file, &prs_anchor, &qt,
                                       palette, palette_size, node_to_palette);
            svg_end_group(&svg_final);

            fprintf(svg_final.file, "<g id=\"NVDR_R1\" opacity=\"0\">\n");
            fprintf(svg_final.file, "<animate attributeName=\"opacity\" from=\"0\" to=\"1\" begin=\"0.2s\" dur=\"0.2s\" fill=\"freeze\"/>\n");
            write_layer(&svg_final, &prs_r1, &qt, palette, palette_size, node_to_palette);
            svg_end_group(&svg_final);

            fprintf(svg_final.file, "<g id=\"NVDR_R2\" opacity=\"0\">\n");
            fprintf(svg_final.file, "<animate attributeName=\"opacity\" from=\"0\" to=\"1\" begin=\"0.4s\" dur=\"0.2s\" fill=\"freeze\"/>\n");
            write_layer(&svg_final, &prs_r2, &qt, palette, palette_size, node_to_palette);
            svg_end_group(&svg_final);

            svg_close(&svg_final);
        }
    }

    // === Export SVBC binary format ===
    char svbc_path[512] = {0};
    if (export_svbc) {
        snprintf(svbc_path, sizeof(svbc_path), "%s", output_path);
        char* dot_svbc = strrchr(svbc_path, '.');
        if (dot_svbc) strcpy(dot_svbc, ".svbc");
        else strcat(svbc_path, ".svbc");
        svbc_write(svbc_path, &qt, img.width, img.height);
    }

    // === Stop timer HERE (before slow PowerShell call) ===
#ifdef _OPENMP
    double t_end = omp_get_wtime();
    double ms_build  = (t_built - t_start) * 1000.0;
    double ms_opt    = (t_opt - t_built) * 1000.0;
    double ms_export = (t_end - t_opt) * 1000.0;
    double elapsed_ms = (t_end - t_start) * 1000.0;
#else
    clock_t t_end = clock();
    double elapsed_ms = (double)(t_end - t_start) / CLOCKS_PER_SEC * 1000.0;
#endif

    long svg_size  = get_file_size(output_path);
    long svbc_size = get_file_size(svbc_path);
    long input_size = get_file_size(input_path);

    // === SVBCZ post-step (optional, outside timer) ===
    char svbcz_path[512] = {0};
    long svbcz_size = 0;
    if (export_svbc && svbc_size > 0) {
        snprintf(svbcz_path, sizeof(svbcz_path), "%s", svbc_path);
        char* dot_ext_svbc = strrchr(svbcz_path, '.');
        if (dot_ext_svbc) strcpy(dot_ext_svbc, ".svbcz");
        else strcat(svbcz_path, ".svbcz");

        char cmd_zip_svbc[2048];
        snprintf(cmd_zip_svbc, sizeof(cmd_zip_svbc),
            "powershell -NoProfile -Command \""
            "$in=[System.IO.File]::ReadAllBytes('%s');"
            "$ms=New-Object System.IO.MemoryStream;"
            "$gz=New-Object System.IO.Compression.GZipStream($ms,[System.IO.Compression.CompressionMode]::Compress);"
            "$gz.Write($in,0,$in.Length);$gz.Close();"
            "[System.IO.File]::WriteAllBytes('%s',$ms.ToArray())\"",
            svbc_path, svbcz_path);
        system(cmd_zip_svbc);
        svbcz_size = get_file_size(svbcz_path);
    }

    // === SVGZ post-step (optional, outside timer) ===
    char svgz_path[512] = {0};
    long svgz_size = 0;
    if (export_svg) {
        snprintf(svgz_path, sizeof(svgz_path), "%s", output_path);
        // Replace .svg with .svgz
        char* dot_ext = strrchr(svgz_path, '.');
        if (dot_ext) strcpy(dot_ext, ".svgz");
        else strcat(svgz_path, ".svgz");

        char cmd_zip[2048];
        snprintf(cmd_zip, sizeof(cmd_zip),
            "powershell -NoProfile -Command \""
            "$in=[System.IO.File]::ReadAllBytes('%s');"
            "$ms=New-Object System.IO.MemoryStream;"
            "$gz=New-Object System.IO.Compression.GZipStream($ms,[System.IO.Compression.CompressionMode]::Compress);"
            "$gz.Write($in,0,$in.Length);$gz.Close();"
            "[System.IO.File]::WriteAllBytes('%s',$ms.ToArray())\"",
            output_path, svgz_path);
        system(cmd_zip);
        svgz_size = get_file_size(svgz_path);
    }

    printf("\n>> NVDR UNIFIED CONTAINER <<\n");
    printf("leaves: %d -> %d | total time: %.0fms\n", leaves_before, leaves_after, elapsed_ms);
#ifdef _OPENMP
    printf(">> Performance: build %.0fms, optimize %.0fms, export %.0fms\n", ms_build, ms_opt, ms_export);
#endif
    if (export_svg) {
        printf("SVG:       %ld KB\n", svg_size / 1024);
        if (svgz_size > 0)
            printf("SVGZ:      %ld KB\n", svgz_size / 1024);
    }
    if (export_svbc && svbc_size > 0) {
        printf("SVBC:      %ld KB  (%.1fx menor que SVG)\n",
               svbc_size / 1024, svg_size > 0 ? (double)svg_size / (double)svbc_size : 0.0);
        if (svbcz_size > 0)
            printf("SVBCZ:     %ld KB  (comprimido gzip)\n", svbcz_size / 1024);
    }
    printf("Original:  %ld KB\n", input_size / 1024);
    
    long best;
    if (svbcz_size > 0)                          best = svbcz_size;
    else if (svbc_size > 0)                       best = svbc_size;
    else if (svgz_size > 0 && svgz_size < svg_size) best = svgz_size;
    else                                          best = svg_size;
    printf("Ratio:     %.0f%% of original\n",
           input_size > 0 ? (double)best / (double)input_size * 100.0 : 0.0);

    /* Codebook persistence: write back so the next run sees this run's
     * palette. Atomic save (.tmp + rename) so a crash mid-write leaves
     * the prior DB intact. */
    if (codebook_db_path) {
        int saved = codebook_db_save(&cb_db, codebook_db_path);
        if (saved == 0) {
            printf("Codebook:  %d unique colors persisted to %s\n",
                   codebook_db_size(&cb_db), codebook_db_path);
        } else {
            fprintf(stderr, "warning: failed to persist codebook to '%s'\n",
                    codebook_db_path);
        }
    }
    codebook_db_free(&cb_db);

    free(node_to_palette); free(palette); free(grad_info);
    free(prs_anchor.node_indices); free(prs_r1.node_indices); free(prs_r2.node_indices);
    quadtree_free(&qt); image_free(&img);
    return 0;
}
