# exports/kittentts/build.mk — KittenTTS text-to-speech export

ENABLE_AUDIO = 1

LTO_SRCS += exports/kittentts/kittentts.c
LTO_SRCS += exports/kittentts/plbert.c
LTO_SRCS += exports/kittentts/text_encoder.c
LTO_SRCS += exports/kittentts/prosody.c
LTO_SRCS += exports/kittentts/acoustic_decoder.c
LTO_SRCS += exports/kittentts/vocoder.c
LTO_SRCS += exports/kittentts/phonemizer.c

CFLAGS += -DENABLE_KITTENTTS -Iexports/kittentts

# APP=1: build standalone binary (links main.c against libsmol.a)
ifeq ($(APP),1)
APP_TARGETS += kittentts

kittentts: libsmol.a exports/kittentts/main.c
	$(CC) $(CFLAGS) -o $@ exports/kittentts/main.c -L. -lsmol $(LDFLAGS)
endif
