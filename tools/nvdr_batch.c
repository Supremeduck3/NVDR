/*
 * nvdr_batch — every image in a folder, coded.
 *
 *   nvdr_batch <folder> [out-folder] [options]
 *
 * Each image (JPEG, PNG, BMP, TGA, GIF, PPM; not a PNG that already carries
 * an NVDR file) becomes out-folder/<name>.nvdr,
 * out-folder being <folder>/nvdr unless given; with -r the folders under
 * it too, in the same tree. Images are coded directionally (the smaller
 * files; --progressive for the layered ones), one per core at once.
 *
 * A run can be stopped and started again: an image whose output is newer
 * than it is skipped (--force codes it anyway), and every file is written
 * under a temporary name and renamed when whole, so a stopped run never
 * leaves half a file that looks done.
 *
 * Each image is reported on a line of its own as it finishes,
 *
 *   [3/120] viagem/praia.jpg  3.1 MB -> 214.0 KB (6.8%)  4.2 s
 *
 * and the page's batch mode (server.js, /batch) reads these lines.
 */
#include "nvdr.h"
#include "pngwrap.h"

#include <dirent.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef _WIN32
#include <direct.h>
#define make_dir(p) _mkdir(p)
#else
#define make_dir(p) mkdir((p), 0755)
#endif

typedef struct {
    char* rel;      /* the path under the folder, '/' separated */
    char* out;      /* the path under the out-folder, without extension */
} Item;

typedef struct {
    Item* items;
    int   n, cap;
} List;

static const char* const EXTS[] = { ".jpg", ".jpeg", ".png", ".bmp", ".tga", ".gif", ".ppm", ".pnm", NULL };

static int is_image(const char* name) {
    const char* dot = strrchr(name, '.');
    if (!dot) return 0;
    for (int i = 0; EXTS[i]; i++) {
        const char *a = dot, *b = EXTS[i];
        while (*a && *b && (*a | 32) == *b) { a++; b++; }
        if (!*a && !*b) return 1;
    }
    return 0;
}

/* strdup(), which C11 does not have. */
static char* dup(const char* a) {
    size_t n = strlen(a) + 1;
    char* s = (char*)malloc(n);
    if (s) memcpy(s, a, n);
    return s;
}

static char* join(const char* a, const char* b) {
    size_t la = strlen(a), lb = strlen(b);
    char* s = (char*)malloc(la + lb + 2);
    if (!s) return NULL;
    memcpy(s, a, la);
    size_t k = la;
    if (la && a[la - 1] != '/' && lb) s[k++] = '/';
    memcpy(s + k, b, lb + 1);
    return s;
}

/* A PNG that nvdr_png or this tool made: an NVDR file already, which
 * coding again would only make worse. Its chunk is always at byte 33. */
static int carries_nvdr(const char* path) {
    uint8_t h[41];
    FILE* f = fopen(path, "rb");
    if (!f) return 0;
    size_t n = fread(h, 1, sizeof h, f);
    fclose(f);
    return n == sizeof h && h[0] == 0x89 && !memcmp(h + 1, "PNG", 3) && !memcmp(h + 37, "nvDr", 4);
}

static int is_dir(const char* path) {
    struct stat st;
    return stat(path, &st) == 0 && S_ISDIR(st.st_mode);
}

/* The folder's images, and with `deep` its folders' (but not `skip`, the
 * out-folder when it is inside), as paths relative to `root`. */
static int walk(const char* root, const char* rel, const char* skip, int deep, List* L) {
    char* dir = rel[0] ? join(root, rel) : dup(root);
    if (!dir) return -1;
    DIR* d = opendir(dir);
    if (!d) { free(dir); return -1; }
    struct dirent* e;
    while ((e = readdir(d))) {
        if (e->d_name[0] == '.') continue;
        char* full = join(dir, e->d_name);
        char* r = rel[0] ? join(rel, e->d_name) : dup(e->d_name);
        if (!full || !r) { free(full); free(r); break; }
        if (is_dir(full)) {
            if (deep && strcmp(full, skip) != 0) walk(root, r, skip, deep, L);
            free(r);
        } else if (is_image(e->d_name) && !carries_nvdr(full)) {
            if (L->n == L->cap) {
                L->cap = L->cap ? 2 * L->cap : 64;
                Item* grown = (Item*)realloc(L->items, (size_t)L->cap * sizeof(Item));
                if (!grown) { free(full); free(r); break; }
                L->items = grown;
            }
            L->items[L->n].rel = r;
            L->items[L->n].out = NULL;
            L->n++;
        } else free(r);
        free(full);
    }
    closedir(d);
    free(dir);
    return 0;
}

static int cmp_item(const void* a, const void* b) {
    return strcmp(((const Item*)a)->rel, ((const Item*)b)->rel);
}

/* rel without its extension: "a/foto.jpg" -> "a/foto". */
static char* stem(const char* rel) {
    char* s = dup(rel);
    if (!s) return NULL;
    char *dot = strrchr(s, '.'), *slash = strrchr(s, '/');
    if (dot && (!slash || dot > slash)) *dot = 0;
    return s;
}

/* Every folder on the way to `path` (a file's path), made. */
static void make_parents(const char* path) {
    char* p = dup(path);
    if (!p) return;
    for (char* c = p + 1; *c; c++) {
        if (*c != '/') continue;
        *c = 0;
        make_dir(p);
        *c = '/';
    }
    free(p);
}

static int newer(const char* out, const char* in) {
    struct stat a, b;
    return stat(out, &a) == 0 && stat(in, &b) == 0 && a.st_mtime >= b.st_mtime;
}

/* Writes the bytes under a temporary name, then renames them into place. */
static int write_whole(const char* path, const uint8_t* data, size_t len) {
    size_t n = strlen(path);
    char* tmp = (char*)malloc(n + 6);
    if (!tmp) return -1;
    memcpy(tmp, path, n);
    memcpy(tmp + n, ".part", 6);
    FILE* f = fopen(tmp, "wb");
    int ok = f && fwrite(data, 1, len, f) == len;
    if (f) ok = fclose(f) == 0 && ok;
    remove(path);    /* rename() does not replace a file everywhere */
    if (ok) ok = rename(tmp, path) == 0;
    if (!ok) remove(tmp);
    free(tmp);
    return ok ? 0 : -1;
}

static void human(double b, char* s, size_t cap) {
    if (b >= 1048576) snprintf(s, cap, "%.1f MB", b / 1048576);
    else if (b >= 1024) snprintf(s, cap, "%.1f KB", b / 1024);
    else snprintf(s, cap, "%.0f B", b);
}

static double now(void) {
#ifdef _OPENMP
    return omp_get_wtime();
#else
    return (double)clock() / CLOCKS_PER_SEC;
#endif
}

static void usage(const char* a0) {
    fprintf(stderr,
        "usage: %s <folder> [out-folder] [options]\n"
        "  every JPEG, PNG, BMP, TGA, GIF and PPM in <folder> to out-folder/<name>.nvdr\n"
        "  (out-folder: <folder>/nvdr unless given)\n"
        "  -r, --recursive  the folders under it too, in the same tree\n"
        "  --as F           nvdr (default), png (the PNG carrying the NVDR, as\n"
        "                   nvdr_png makes it) or both\n"
        "  --jobs N         images coded at once (default: one per core)\n"
        "  --force          code images whose output is already newer than them\n"
        "  --progressive    the layered mode instead of the directional one\n"
        "  --q N            quantiser step (default 24; smaller is better and larger)\n"
        "  --fast           try the other transform types only on the two cheapest\n"
        "                   modes of a leaf, ranked without RDOQ: about 3.5 times\n"
        "                   faster, ~1%% bigger\n"
        "  --grain M        off (default), auto or on, as nvdr_encode's\n"
        "  --chroma M       auto (default), 420 or 444, as nvdr_encode's\n", a0);
}

int main(int argc, char** argv) {
    if (argc < 2 || argv[1][0] == '-') { usage(argv[0]); return 2; }
    const char* root = argv[1];
    const char* out_root = NULL;
    int deep = 0, force = 0, jobs = 0, as_nvdr = 1, as_png = 0;
    NvdrConfig cfg = nvdr_default_config();
    cfg.directional = 1;
    int first = 2;
    if (argc > 2 && argv[2][0] != '-') { out_root = argv[2]; first = 3; }
    for (int i = first; i < argc; i++) {
        if (!strcmp(argv[i], "-r") || !strcmp(argv[i], "--recursive")) deep = 1;
        else if (!strcmp(argv[i], "--force")) force = 1;
        else if (!strcmp(argv[i], "--jobs") && i + 1 < argc) jobs = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--progressive")) cfg.directional = 0;
        else if (!strcmp(argv[i], "--q") && i + 1 < argc) cfg.q = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--fast")) { cfg.tx_modes = 2; cfg.fast_rank = 1; }
        else if (!strcmp(argv[i], "--as") && i + 1 < argc) {
            const char* v = argv[++i];
            as_nvdr = !strcmp(v, "nvdr") || !strcmp(v, "both");
            as_png = !strcmp(v, "png") || !strcmp(v, "both");
            if (!as_nvdr && !as_png) { fprintf(stderr, "--as takes nvdr, png or both\n"); return 2; }
        } else if (!strcmp(argv[i], "--grain") && i + 1 < argc) {
            const char* v = argv[++i];
            if (!strcmp(v, "off")) cfg.grain = NVDR_GRAIN_OFF;
            else if (!strcmp(v, "auto")) cfg.grain = NVDR_GRAIN_AUTO;
            else if (!strcmp(v, "on")) cfg.grain = NVDR_GRAIN_ON;
            else { fprintf(stderr, "--grain takes off, auto or on\n"); return 2; }
        } else if (!strcmp(argv[i], "--chroma") && i + 1 < argc) {
            const char* v = argv[++i];
            if (!strcmp(v, "420")) cfg.chroma420 = NVDR_CHROMA_420;
            else if (!strcmp(v, "444")) cfg.chroma420 = NVDR_CHROMA_444;
            else if (!strcmp(v, "auto")) cfg.chroma420 = NVDR_CHROMA_AUTO;
            else { fprintf(stderr, "--chroma takes auto, 420 or 444\n"); return 2; }
        } else { fprintf(stderr, "unknown option '%s'\n", argv[i]); usage(argv[0]); return 2; }
    }
    if (!is_dir(root)) { fprintf(stderr, "'%s' is not a folder\n", root); return 1; }
    char* def_out = out_root ? NULL : join(root, "nvdr");
    if (!out_root) out_root = def_out;
    make_dir(out_root);
    if (!is_dir(out_root)) { fprintf(stderr, "cannot make the folder '%s'\n", out_root); return 1; }

    List L = { NULL, 0, 0 };
    walk(root, "", out_root, deep, &L);
    if (L.n == 0) { printf("nenhuma imagem em %s\n", root); free(def_out); return 0; }
    qsort(L.items, (size_t)L.n, sizeof(Item), cmp_item);
    /* foto.jpg and foto.png in one folder would both be foto.nvdr: such
     * names keep their extension (foto.jpg.nvdr). */
    for (int i = 0; i < L.n; i++) {
        char* s = stem(L.items[i].rel);
        int clash = 0;
        for (int j = 0; j < L.n && !clash; j++) {
            if (j == i) continue;
            char* t = stem(L.items[j].rel);
            clash = t && s && !strcmp(s, t);
            free(t);
        }
        char* base = clash ? dup(L.items[i].rel) : s;
        if (clash) free(s);
        L.items[i].out = base ? join(out_root, base) : NULL;
        free(base);
    }

#ifdef _OPENMP
    if (jobs <= 0) jobs = omp_get_num_procs();
#endif
    if (jobs <= 0) jobs = 1;
    printf("%d imagens de %s para %s, %d por vez, modo %s, q %d\n", L.n, root, out_root, jobs,
           cfg.directional ? "direcional" : "progressivo", cfg.q);
    fflush(stdout);

    double in_total = 0, out_total = 0, t0 = now();
    int done = 0, skipped = 0, failed = 0;
#pragma omp parallel for schedule(dynamic, 1) num_threads(jobs)
    for (int i = 0; i < L.n; i++) {
        Item* it = &L.items[i];
        char* in = join(root, it->rel);
        char *nv = it->out ? join(it->out, "") : NULL, *pn = NULL;
        if (nv) {
            size_t n = strlen(nv);
            char* a = (char*)malloc(n + 6);
            char* b = (char*)malloc(n + 5);
            if (a) { memcpy(a, nv, n); memcpy(a + n, ".nvdr", 6); }
            if (b) { memcpy(b, nv, n); memcpy(b + n, ".png", 5); }
            free(nv);
            nv = a; pn = b;
        }
        char msg[512] = "";
        int skip = 0, ok = 0;
        double ib = 0, ob = 0, t = now();
        if (!in || !nv || !pn) snprintf(msg, sizeof msg, "ERRO: sem memória");
        else if (!force && (!as_nvdr || newer(nv, in)) && (!as_png || newer(pn, in))) skip = 1;
        else {
            NvdrImage img;
            uint8_t* buf = NULL;
            size_t len = 0;
            struct stat st;
            if (stat(in, &st) == 0) ib = (double)st.st_size;
            if (nvdr_image_load(&img, in) != 0) snprintf(msg, sizeof msg, "ERRO: não consegui ler a imagem");
            else {
                if (nvdr_encode_mem(&buf, &len, &img, &cfg, NULL) != 0)
                    snprintf(msg, sizeof msg, "ERRO: o codificador falhou");
                else {
                    make_parents(nv);
                    ok = 1;
                    if (as_nvdr) {
                        if (write_whole(nv, buf, len) == 0) ob += (double)len;
                        else { ok = 0; snprintf(msg, sizeof msg, "ERRO: não consegui gravar %s", nv); }
                    }
                    if (ok && as_png) {
                        /* The PNG's preview is the picture as decoded. */
                        NvdrImage dec;
                        NvdrHeader h;
                        NvdrDecodeInfo di;
                        NvdrPngResult r;
                        char* tmp = join(pn, "");
                        size_t n = tmp ? strlen(tmp) : 0;
                        char* part = tmp ? (char*)malloc(n + 6) : NULL;
                        if (part) { memcpy(part, tmp, n); memcpy(part + n, ".part", 6); }
                        ok = part && nvdr_decode_mem(buf, len, -1, &dec, &h, &di) == 0;
                        if (ok) {
                            NvdrPngOptions po = nvdr_png_options();
                            ok = nvdr_png_write(part, buf, len, &dec, &po, &r) == 0;
                            nvdr_image_free(&dec);
                            remove(pn);
                            if (ok) ok = rename(part, pn) == 0;
                            if (!ok) remove(part);
                            else ob += (double)r.bytes;
                        }
                        if (!ok) snprintf(msg, sizeof msg, "ERRO: não consegui gravar %s", pn);
                        free(tmp); free(part);
                    }
                    free(buf);
                }
                nvdr_image_free(&img);
            }
        }
#pragma omp critical(batch_report)
        {
            done++;
            char a[32], b[32];
            if (skip) { skipped++; printf("[%d/%d] %s  já feito\n", done, L.n, it->rel); }
            else if (!ok) { failed++; printf("[%d/%d] %s  %s\n", done, L.n, it->rel, msg[0] ? msg : "ERRO"); }
            else {
                in_total += ib; out_total += ob;
                human(ib, a, sizeof a); human(ob, b, sizeof b);
                printf("[%d/%d] %s  %s -> %s (%.1f%%)  %.1f s\n", done, L.n, it->rel, a, b,
                       ib > 0 ? 100.0 * ob / ib : 0.0, now() - t);
            }
            fflush(stdout);
        }
        free(in); free(nv); free(pn);
    }
    char a[32], b[32];
    human(in_total, a, sizeof a); human(out_total, b, sizeof b);
    printf("total: %d codificadas, %s -> %s (%.1f%%) em %.1f s", L.n - skipped - failed, a, b,
           in_total > 0 ? 100.0 * out_total / in_total : 0.0, now() - t0);
    if (skipped) printf(", %d já feitas", skipped);
    if (failed) printf(", %d com erro", failed);
    printf("\n");
    for (int i = 0; i < L.n; i++) { free(L.items[i].rel); free(L.items[i].out); }
    free(L.items);
    free(def_out);
    return failed ? 1 : 0;
}
