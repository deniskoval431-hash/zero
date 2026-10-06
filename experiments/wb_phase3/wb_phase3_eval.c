/*
 * wb_phase3_eval.c -- Warren Braid Phase 3 evaluation harness.
 *
 * Glue only: it #includes the unmodified trainer (../literary_lm.c) with its
 * main() renamed, then runs masked / unmasked teacher-forced evaluation over
 * uint16 token files and emits one summary row per file.
 *
 * Build (MinGW, OpenBLAS):
 *   gcc -O2 -std=c11 -DUSE_OPENBLAS -Dmain=phase3_unused_main \
 *       -I.. wb_phase3_eval.c -o wb_phase3_eval.exe -L.. -lopenblas -lm
 *
 * Usage:
 *   wb_phase3_eval.exe --preset literary --context 512 --vocab 133 \
 *       --tokenizer wb_phase3.vocab --model MODEL.f32 \
 *       --file F.tok --mode text|cond --label armA
 *
 * mode text : unmasked next-token loss over all context windows
 *             (stride = context, non-overlapping).
 * mode cond : channel_loss_mask semantics; loss only from TARGET onward
 *             (symbolic-prefix -> text continuation).
 *
 * Output (stdout, CSV header once):
 *   label,file,mode,window,tokens,nll_mean,top1_acc,copy_tokens,copy_nll_mean,copy_top1_acc,novel_tokens,novel_nll_mean,novel_top1_acc
 * copy = the target token already occurs earlier in the same window.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define main phase3_trainer_main
#include "../../literary_lm.c"
#undef main

static void usage(void)
{
    fprintf(stderr,
            "usage: wb_phase3_eval --tokenizer FILE --model FILE --file FILE "
            "--mode text|cond --label NAME [--preset literary] "
            "[--context N --dim N --heads N --layers N --ff N --vocab N]\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char **argv)
{
    Config cfg;
    Model model;
    Tokenizer tokenizer;
    Corpus file;
    Rng rng;
    const char *model_path = NULL;
    const char *tokenizer_path = NULL;
    const char *file_path = NULL;
    const char *label = "arm?";
    const char *mode = "text";
    const char *preset = "literary";
    int i;
    const char *base;
    int cond;

    cfg = preset_config(preset);
    for (i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--preset") == 0 && i + 1 < argc) {
            preset = argv[++i];
            cfg = preset_config(preset);
        } else if (strcmp(argv[i], "--context") == 0 && i + 1 < argc) {
            cfg.context = (int)parse_long(argv[++i], "--context");
        } else if (strcmp(argv[i], "--dim") == 0 && i + 1 < argc) {
            cfg.dim = (int)parse_long(argv[++i], "--dim");
        } else if (strcmp(argv[i], "--heads") == 0 && i + 1 < argc) {
            cfg.heads = (int)parse_long(argv[++i], "--heads");
        } else if (strcmp(argv[i], "--layers") == 0 && i + 1 < argc) {
            cfg.layers = (int)parse_long(argv[++i], "--layers");
        } else if (strcmp(argv[i], "--ff") == 0 && i + 1 < argc) {
            cfg.ff = (int)parse_long(argv[++i], "--ff");
        } else if (strcmp(argv[i], "--vocab") == 0 && i + 1 < argc) {
            cfg.vocab = (int)parse_long(argv[++i], "--vocab");
        } else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--tokenizer") == 0 && i + 1 < argc) {
            tokenizer_path = argv[++i];
        } else if (strcmp(argv[i], "--file") == 0 && i + 1 < argc) {
            file_path = argv[++i];
        } else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            mode = argv[++i];
        } else if (strcmp(argv[i], "--label") == 0 && i + 1 < argc) {
            label = argv[++i];
        } else {
            usage();
        }
    }
    if (model_path == NULL || tokenizer_path == NULL || file_path == NULL) {
        usage();
    }
    cond = strcmp(mode, "cond") == 0;
    if (!cond && strcmp(mode, "text") != 0) usage();

    memset(&tokenizer, 0, sizeof(tokenizer));
    tokenizer_load(&tokenizer, tokenizer_path);
    if (tokenizer.loaded && tokenizer.vocab != cfg.vocab) {
        fail("tokenizer vocabulary does not match model vocabulary");
    }

    rng_seed(&rng, 1);
    model_create(&model, cfg, &rng);
    checkpoint_load(model_path, &model, &rng);

    memset(&file, 0, sizeof(file));
    corpus_add_file(&file, file_path, tokenizer.loaded ? tokenizer.token_width : 1);
    for (i = 0; i < (int)file.length; ++i) {
        if (file.data[i] >= cfg.vocab) {
            fail("evaluation file contains a token outside the vocabulary");
        }
    }

    base = strrchr(file_path, '/');
    if (base == NULL) base = strrchr(file_path, '\\');
    base = base != NULL ? base + 1 : file_path;

    {
        unsigned char *mask = zero_alloc((size_t)cfg.context, 1);
        double sum_nll = 0.0, sum_copy = 0.0, sum_novel = 0.0;
        long long n = 0, n_copy = 0, n_novel = 0, windows = 0;
        long long top1 = 0, top1_copy = 0, top1_novel = 0;
        size_t start;
        int printed_header = 0;

        for (start = 0; start + (size_t)cfg.context < file.length;
             start += (size_t)cfg.context) {
            const Token *window = file.data + start;
            const Token *targets = window + 1;
            const unsigned char *active = NULL;
            int time;

            if (cond) {
                if (channel_loss_mask(window, window, mask, cfg.context) == 0) {
                    continue; /* no reply target inside this window */
                }
                active = mask;
            }
            model_forward_masked(&model, window, targets, 0.0f, NULL, active);
            ++windows;
            for (time = 0; time < cfg.context; ++time) {
                float probability;
                int target, argmax, tok;
                int is_copy = 0;
                if (active != NULL && !active[time]) continue;
                target = targets[time];
                probability = model.probs[(size_t)time * cfg.vocab + target];
                if (probability < 1.0e-20f) probability = 1.0e-20f;
                argmax = 0;
                for (tok = 1; tok < cfg.vocab; ++tok) {
                    if (model.probs[(size_t)time * cfg.vocab + tok] >
                        model.probs[(size_t)time * cfg.vocab + argmax]) {
                        argmax = tok;
                    }
                }
                for (tok = 0; tok < time; ++tok) {
                    if (window[tok] == (Token)target) {
                        is_copy = 1;
                        break;
                    }
                }
                sum_nll += -log((double)probability);
                ++n;
                if (argmax == target) ++top1;
                if (is_copy) {
                    sum_copy += -log((double)probability);
                    ++n_copy;
                    if (argmax == target) ++top1_copy;
                } else {
                    sum_novel += -log((double)probability);
                    ++n_novel;
                    if (argmax == target) ++top1_novel;
                }
            }
            if (!printed_header) {
                printf("label,file,mode,window,tokens,nll_mean,top1_acc,"
                       "copy_tokens,copy_nll_mean,copy_top1_acc,"
                       "novel_tokens,novel_nll_mean,novel_top1_acc\n");
                printed_header = 1;
            }
        }
        if (n == 0) {
            fprintf(stderr, "warning: %s: no evaluable positions (%s mode)\n",
                    file_path, mode);
            return EXIT_FAILURE;
        }
        double copy_nll = n_copy ? sum_copy / n_copy : 0.0;
        double novel_nll = n_novel ? sum_novel / n_novel : 0.0;
        printf("%s,%s,%s,%lld,%lld,%.6f,%.6f,%lld,%.6f,%.6f,%lld,%.6f,%.6f\n",
               label, base, mode,
               windows,
               n,
               sum_nll / n, n ? (double)top1 / n : 0.0,
               n_copy,
               copy_nll, n_copy ? (double)top1_copy / n_copy : 0.0,
               n_novel,
               novel_nll, n_novel ? (double)top1_novel / n_novel : 0.0);
        free(mask);
    }

    corpus_destroy(&file);
    model_destroy(&model);
    return EXIT_SUCCESS;
}
