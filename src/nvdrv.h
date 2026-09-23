/*
 * NVDRV — a sequence of NVDR frames.
 *
 * The still format already says "this level is a delta on the one before
 * it". A sequence says the same thing along time: frame N is a delta on
 * what the decoder is holding after frame N-1. Nothing about the frame
 * codec changes — a predicted frame is an ordinary NVDR container whose
 * image happens to be the prediction error, biased to the middle of the
 * range so it fits in the same unsigned bytes.
 *
 * WHY NOT COPY
 * ------------
 * The obvious design is to mark regions that have not changed and copy
 * them. Measured, that is worth almost nothing: between two *identical*
 * frames only 38% of the area can be copied, because the reference is the
 * previous frame decoded, and at the rates this codec runs at that
 * reconstruction already differs from the source by more than the
 * tolerance across most of the picture. Under a 3 px pan it falls to 0.5%.
 * Coding the residual instead is worth 65 to 70%.
 *
 * MOTION IS NOT OPTIONAL
 * ----------------------
 * A co-located reference is worth 7% once the camera moves at all. One
 * global translation, the crudest model there is, takes the same sequence
 * to 61% and raises quality by 0.97 dB. Per-block vectors would do better
 * and are not here yet.
 *
 * WHAT THE RECONSTRUCTION LOOP IS FOR
 * -----------------------------------
 * The encoder predicts from its own decoded output, never from the source
 * frame, because that is all the decoder will have. Getting this wrong is
 * how a codec drifts: the two sides diverge a little per frame until the
 * picture is wrong. Measured over an eight-frame chain there is no drift —
 * quality climbs 1.89 dB while cost per frame falls to a fifth, because
 * each residual adds detail on top of what the last one left. A shot that
 * holds still gets better and cheaper the longer it runs.
 *
 * TRUNCATION, ON BOTH AXES
 * ------------------------
 * Frames are self-delimiting and carry no index, so a prefix of the file
 * is a prefix of the movie. The last frame in a cut file is handed to the
 * still decoder as-is, which decodes as far as its bytes reach. Cutting
 * anywhere gives whole frames, then one partial one, then nothing.
 */
#ifndef NVDRV_H
#define NVDRV_H

#include "nvdr.h"

#define NVDRV_MAGIC        "NVDV"
#define NVDRV_VERSION      6
#define NVDRV_HEADER_SIZE  24
#define NVDRV_FRAME_HEADER 12

#define NVDRV_INTRA 0   /* the frame on its own */
#define NVDRV_PRED  1   /* the error against the previous reconstruction */

typedef struct {
    NvdrConfig frame;      /* how each frame's container is built */
    /*
     * Force an intra frame every N, so the stream can be joined partway
     * and a decode error cannot poison the rest of the movie. 0 means only
     * frame 0 is intra, which is smaller and unseekable.
     */
    int   gop;
    /*
     * Half-width of the global motion search, in pixels. 0 pins the
     * reference in place, which measured worthless under a 3 px pan.
     */
    int   search;
    /*
     * Mean absolute prediction error, in levels, above which a frame is
     * coded intra even if the GOP does not ask for it. The bias to 128
     * clips anything past +/-127, so a scene cut has to be caught rather
     * than coded badly.
     */
    float intra_threshold;
    /*
     * Block size for per-block motion, in pixels. 0 keeps one global
     * vector per frame. A block's vector is searched around the global one
     * and around zero, and sent as a delta against the global one, so a
     * shot with nothing moving against the camera costs almost nothing
     * extra for having it on.
     */
    int   block;
    /* Quantiser step for predicted frames; zero means 1.2 times the intra
     * frames' `frame.q`. */
    int   pred_q;
    /* What one bit of motion field is worth in sum of absolute differences
     * over a block, when the encoder chooses between a block's own vector
     * and the one its neighbours predict. 0 keeps the search's choice. */
    int   mv_lambda;
    int   fps;             /* carried in the header, informational */
} NvdrvConfig;

NvdrvConfig nvdrv_default_config(void);

/* ------------------------------------------------------------- encoder */

/*
 * Streaming, because a movie does not fit in memory. Frames go in one at
 * a time and the container is written as they arrive.
 */
typedef struct NvdrvEncoder NvdrvEncoder;

int  nvdrv_encode_open(NvdrvEncoder** out, const char* path,
                       int width, int height, const NvdrvConfig* cfg);
/* Returns 0 on success. `kind_out` and `bytes_out` may be NULL; when
 * given they report how the frame was coded and what it cost. */
int  nvdrv_encode_frame(NvdrvEncoder* enc, const NvdrImage* frame,
                        int* kind_out, size_t* bytes_out,
                        int* dx_out, int* dy_out);
int  nvdrv_encode_close(NvdrvEncoder* enc);

/* ------------------------------------------------------------- decoder */

typedef struct {
    int width, height;
    int frame_count;     /* what the header claims; a cut file has fewer */
    int fps, gop;
} NvdrvInfo;

typedef struct NvdrvDecoder NvdrvDecoder;

int  nvdrv_decode_open(NvdrvDecoder** out, const char* path, NvdrvInfo* info);

/*
 * Render the next frame into `out`, which the caller allocates at
 * width*height*3. Returns 1 when a frame was produced, 0 at the end of the
 * stream, -1 on a container that does not make sense.
 *
 * A partial frame at the end of a cut file still produces a picture, at
 * whatever quality its bytes paid for.
 */
int  nvdrv_decode_next(NvdrvDecoder* dec, NvdrImage* out,
                       int* kind_out, int* partial_out);
void nvdrv_decode_close(NvdrvDecoder* dec);

/* ------------------------------------------------- one image from another */

/*
 * One image predicted from another of the same size, outside a sequence:
 * what an album uses for a photo that repeats the one before it. `recon`
 * (allocated) is what the decoder will show. `cfg->pred_q` > 0 sets the
 * residual's step, otherwise `cfg->frame.q`.
 */
int nvdrv_predict_encode(const NvdrImage* ref, const NvdrImage* cur, const NvdrvConfig* cfg,
                         uint8_t** out, size_t* out_len, NvdrImage* recon);

/*
 * The same in two steps, for an encoder that codes one prediction at
 * several steps: the motion search, which does not depend on the step and
 * is half the work, once; then the residual at any step. The residual
 * step returns 1, with nothing allocated, as soon as the payload is known
 * to reach `limit` bytes, before decoding it back.
 */
typedef struct {
    NvdrvConfig cfg;
    int         block, dx, dy;
    uint8_t*    field;       /* the packed motion field */
    size_t      field_len;
    NvdrImage   err;         /* current - prediction + 128 */
} NvdrvMotion;

int  nvdrv_motion_find(const NvdrImage* ref, const NvdrImage* cur, const NvdrvConfig* cfg,
                       NvdrvMotion* m);
int  nvdrv_motion_encode(const NvdrvMotion* m, const NvdrImage* ref, int q, size_t limit,
                         uint8_t** out, size_t* out_len, NvdrImage* recon);
void nvdrv_motion_free(NvdrvMotion* m);
int nvdrv_predict_decode(const NvdrImage* ref, const uint8_t* data, size_t len,
                         NvdrImage* out, int* partial);

#endif /* NVDRV_H */
