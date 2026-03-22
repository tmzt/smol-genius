/*
 * image.h - Generic image loading, resize, and normalization
 */

#ifndef SMOL_IMAGE_H
#define SMOL_IMAGE_H

/* Load image via stb_image, resize to target_size x target_size,
 * normalize to [-1, 1], return channel-first [3, H, W] float array.
 * Caller must free. Sets *out_w and *out_h to target_size. */
float *smol_load_image(const char *path, int target_size, int *out_w, int *out_h);

#endif /* SMOL_IMAGE_H */
