# F1 Facts Pipeline

A small, real, end to end content generation pipeline: give it a topic, it researches facts, writes a script, generates narration, and renders a finished vertical video with synced captions, ready for Shorts, Reels, or TikTok.

```
topic
  │
  ▼
research.py     →  pull verified facts for the topic
  │
  ▼
script_gen.py   →  turn facts into a 30 to 45 second spoken script
  │
  ▼
tts.py          →  render each line to narration audio
  │
  ▼
video.py        →  render caption cards, sync to audio, stitch into an mp4
  │
  ▼
metadata.py     →  generate title, caption, hashtags
  │
  ▼
finished clip + metadata
```

## Why it's built this way

Every stage sits behind a small interface (`FactSource`, `ScriptGenerator`, `VoiceSynth`) instead of a concrete class. `pipeline.py` is the only file that wires a specific implementation to each stage, and `config.py` picks which implementation from environment variables, so swapping the curated fact set for a live API, or the offline voice for a free neural one, is a shell variable, not a code change, nothing downstream has to change or even know. See "Connecting the free upgrades" below.

That's not decoration, it's a direct answer to two real constraints hit while building this:

- The sandbox this was first built in only allows network access to package registries, so the live F1 API and Hugging Face (where better voice models live) are both unreachable from here.
- Not every viewer of this repo will have an Anthropic or ElevenLabs key on hand.

Rather than fake those integrations or block on API access, the default path for every stage is something that actually runs today with nothing to sign up for, and the paid or live path is written, documented, and ready to drop in.

| Stage | Default (works with nothing) | Upgrade path |
|---|---|---|
| Research | `StaticF1FactSource`, curated JSON in `pipeline/data/f1_facts.json` | `LiveF1ApiFactSource`, Jolpica F1 REST API, no key, needs open network |
| Script | `TemplateScriptGenerator`, deterministic | `OllamaScriptGenerator`, real local LLM, free, no key. `AnthropicScriptGenerator`, real Claude call, needs `ANTHROPIC_API_KEY` and costs per call |
| Voice | `EspeakVoice`, local formant synthesis via `espeak-ng` | `PiperVoice`, local neural TTS, needs a downloaded `.onnx` model. `ElevenLabsVoice` stub, needs `ELEVENLABS_API_KEY` |
| Video | `SolidBackground`, a flat color caption card rendered with Pillow, stitched with ffmpeg | `VoxelDropBackground`, an original animated block field, same interface, drop-in via `PIPELINE_BACKGROUND=voxel` |
| Audio | Narration only | Background music mixed in via `PIPELINE_MUSIC_PATH`, see "Adding background music" below |
| Metadata | Template based title, caption, hashtags | Same reasoning as script gen: this doesn't need an LLM to do a good job |

One deliberate choice worth flagging: the video stays generated, not sourced. No stock footage, no scraped "no copyright" reuploads, no generated likenesses of real drivers. It would be easy to drop in a real Minecraft parkour clip behind the captions, that's a common format for a reason, it holds attention. It's also someone else's copyrighted asset, and "no copyright" in a video title doesn't actually mean that, most of those reuploads are themselves infringing. `VoxelDropBackground` (see below) is the honest version of the same idea: animated, colorful, something moving behind the text, generated from scratch with Pillow instead of sourced from anywhere, so there's nothing to clear.

## Connecting the free upgrades

Every stage has a better version that costs nothing, just needs open network access this sandbox doesn't have. Nothing below needs a code change, it's all environment variables, that's what `Pipeline.from_env()` in `pipeline/pipeline.py` and `pipeline/config.py` are for.

**Live facts instead of the curated set.** Uses the Jolpica F1 API, free, no key.

```bash
export PIPELINE_FACTS=live
python3 cli.py --topic max_verstappen-2024   # topic is "<driverId>-<season>", see the gotcha below
```

Verify it works on your machine first, independent of this project:

```bash
curl https://api.jolpi.ca/ergast/f1/2024/drivers/max_verstappen/driverStandings.json
```

Note the driverId: it's `max_verstappen`, not `verstappen`, disambiguated against his father Jos Verstappen who also raced in F1. Most drivers are just their lowercase surname (`norris`, `leclerc`), some aren't, look one up with `curl https://api.jolpi.ca/ergast/f1/2024/drivers.json` if unsure.

If that doesn't come back looking like the shape `LiveF1ApiFactSource._parse_standings` expects in `pipeline/research.py`, that function is the only place to fix.

**A real local LLM for the script instead of the template, through Ollama, the same tool you already run for the F1 discovery platform and AuraOS.** Zero cost, it's your own hardware doing the inference.

```bash
ollama pull llama3.1        # once, or use a model you already have, e.g. llama3, gemma2
export PIPELINE_SCRIPT=ollama
export OLLAMA_MODEL=llama3.1   # match whatever you actually pulled
python3 cli.py --topic senna
```

**Neural voice instead of the robotic default, through Piper, also free, just a one time model download.**

```bash
mkdir -p voices
curl -L -o voices/en_US-lessac-medium.onnx \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
curl -L -o voices/en_US-lessac-medium.onnx.json \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json

export PIPELINE_VOICE=piper
export PIPER_MODEL_PATH=voices/en_US-lessac-medium.onnx
python3 cli.py --topic senna
```

Stack all three and the whole pipeline runs on real live data, a real local LLM, and a real neural voice, without a single dollar spent or a single API key signed up for:

```bash
export PIPELINE_FACTS=live PIPELINE_SCRIPT=ollama PIPELINE_VOICE=piper PIPER_MODEL_PATH=voices/en_US-lessac-medium.onnx
python3 cli.py --topic max_verstappen-2024
```

All three have now been run for real, on real hardware, not just documented. Two real bugs turned up in the process, both fixed:

- The live fact source's first version assumed every driver's `driverId` was their lowercase surname. Verstappen's is `max_verstappen`, disambiguated against his father Jos Verstappen who also raced in F1. Caught by actually calling the API, not by guessing.
- The Ollama script generator went through two failed prompt formats before landing on the real fix. A `---` delimited format broke because llama3 prepended chatty preamble before the content. Switching to `HOOK:`/`BODY:`/`CTA:` labels with a parser that ignores anything before them broke differently on the very next run, llama3 skipped the labels entirely and just wrote free prose. The actual fix was Ollama's `format` field, which accepts a JSON schema and constrains decoding to match it, so the model structurally cannot return something that fails to parse. Claude, by contrast, followed a plain text format instruction correctly every single time in this project, that gap between a frontier model and an 8B local one is worth remembering for the DSA and system design conversations too.

The paid options, `AnthropicScriptGenerator` and the `ElevenLabsVoice` stub, are still there in the code if you ever want a quality ceiling above the free stack, but they cost real money per call, they're not part of the zero cost path.

**An animated background instead of the flat card.** No download, no key, it's generated at render time.

```bash
export PIPELINE_BACKGROUND=voxel
python3 cli.py --topic senna
```

`VoxelDropBackground` (`pipeline/video.py`) renders a field of colored blocks that drop in and settle into a skyline, seeded per topic so each clip's layout is different but reproducible on rerun. It's drawn frame by frame with Pillow, a dark overlay is blended on top so the caption text stays readable, and ffmpeg encodes the frame sequence exactly the same way it already encoded the single static frame, `assemble_video` doesn't know or care which one it got, that's the same interface pattern as every other stage.

**Background music instead of narration alone.** This is the one upgrade that isn't zero cost to source, even when it's free to use, because a real music file, licensed or not, isn't something to bundle silently into a git repo. Grab one first:

- YouTube Audio Library (studio.youtube.com → Audio Library) or Pixabay Music are the easiest, no attribution required on most tracks.
- incompetech.com (Kevin MacLeod) is CC-BY, free to use, attribution required in the video description if you actually post it.

Then:

```bash
export PIPELINE_MUSIC_PATH=/path/to/track.mp3
python3 cli.py --topic senna
```

`assemble_video` builds the captioned video exactly as before, then runs one more ffmpeg pass that loops the track if it's shorter than the clip, ducks it to 12% volume so it sits under the narration instead of competing with it, and mixes rather than replaces the narration track. No track ships with this repo; `PIPELINE_MUSIC_PATH` unset means silent background music, same as before this feature existed.

Stack everything:

```bash
export PIPELINE_FACTS=live PIPELINE_SCRIPT=ollama PIPELINE_VOICE=piper PIPER_MODEL_PATH=voices/en_US-lessac-medium.onnx PIPELINE_BACKGROUND=voxel PIPELINE_MUSIC_PATH=/path/to/track.mp3
python3 cli.py --topic max_verstappen-2024
```

## Setup

```bash
pip install -r requirements.txt
sudo apt-get install -y espeak-ng   # local voice synthesis, no key required
```

## Run it

```bash
python3 cli.py --list-topics
python3 cli.py --topic senna
```

Output lands in `outputs/<topic>.mp4`, plus a title, caption, and hashtags printed to the terminal.

Available topics ship in `pipeline/data/f1_facts.json`: `senna`, `schumacher`, `monaco-gp`, `hamilton-verstappen-2021`, `ground-effect-2022`. Add a new topic by adding an entry to that file, no code changes needed.

## Tests

```bash
pip install pytest
python3 -m pytest tests/ -v
```

Tests cover the pure logic stages (research, script generation, metadata). The subprocess backed stages (`tts.py`, `video.py`) are exercised by actually running the CLI rather than mocked, since the point of this project is that the pipeline really produces a real file, not that it appears to.

## Honest limitations

- Both the zero cost/zero setup default and the full free upgrade path (`live` facts, `ollama` script, `piper` voice, `voxel` background, mixed music) have now been run end to end for real and have committed output to show for it, not just documentation.
- `VoxelDropBackground` is intentionally simple, dropping blocks and a small idle bounce, not a physics engine or a game. Making it fancier is a real next step, not done here, because the point of this project was proving the pattern end to end, not maximizing watch time.
- `LiveF1ApiFactSource` is scoped to one endpoint, season standings, which gives two solid facts, not five. Richer facts (race by race results, qualifying) are a natural next step, not done here.
- The Ollama backed script quality depends on the local model. `llama3:latest` (8B) needed schema constrained decoding to be reliable at all, a bigger local model or Claude through `AnthropicScriptGenerator` will write a sharper hook.
- This produces one finished clip per run for one niche. It intentionally does not attempt auto posting, analytics, or running multiple niches at once, that's a much bigger and riskier system than a five day build should try to prove out in one shot. See the planning notes this project grew out of for why.

## What this is meant to demonstrate

A working multi stage AI pipeline: real API and subprocess integration, a clean provider abstraction that was actually exercised by swapping constraints during the build (network blocked, no paid keys), and honest handling of the parts that could not be verified in this environment rather than papering over them.
