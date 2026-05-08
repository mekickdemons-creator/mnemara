# Sentinel Demo Screencast — 90 seconds

Goal: show one concrete moment — an agent halting itself instead of
burning through API budget on a runaway loop. Single terminal, no
editing, no narration overlay (subtitles only if you want).

## Setup before recording

1. Fresh terminal, large font (>= 16pt), dark background.
2. Working dir: `~/demo`. Empty.
3. Have an `ANTHROPIC_API_KEY` in env.
4. Pre-stage a file that will trip Sentinel's polling rule:
   `echo "TODO: this never resolves" > ~/demo/.waiting_signal`

## Shot list

```
[0:00–0:08]   Title card / blank terminal with one line:
              "Mnemara + Sentinel — an agent that halts itself"
              Hold for ~3s. Then start typing.

[0:08–0:20]   Type and run:
                  pip install mnemara
              Cut output to 2-3 lines (or use a pre-installed venv).

[0:20–0:35]   Type and run:
                  mnemara init --instance demo
                  mnemara role --instance demo --set-from-url \
                    https://raw.githubusercontent.com/mekickdemons-creator/mnemara/main/examples/roles/sentinel.md
              Show the "downloaded" + "role doc set" lines clearly.

[0:35–0:42]   Type and run:
                  mnemara run --instance demo

[0:42–1:00]   At the prompt, paste this in:
                  Watch ~/demo/.waiting_signal and tell me when its
                  contents change. Don't stop until it changes.

              Let the agent run. It will read the file, see
              "TODO: this never resolves", read again, read again...

[1:00–1:20]   ON THE 3RD READ: cut to a close-up (or just zoom the
              terminal) showing the Sentinel halt:

                  [SENTINEL HALT — POLLING]
                  tool: Read
                  args: ~/demo/.waiting_signal
                  calls so far: 3
                  what changed between calls: nothing
                  proposed next step: ask user...

[1:20–1:30]   Closing card / overlay:
                  "The agent stopped itself.
                   Edit the rules. They run every turn.
                   pip install mnemara"
              Show the GH URL.
```

## Post

- Export 1080p, mp4.
- Upload to YouTube unlisted, embed in a DEV.to post titled:
  **"I made an agent that halts itself before burning through my API
  budget."**
- Cross-post the GH URL + 90s clip in:
  - Existing DEV.to thread (reply)
  - Mnemara GH Discussions (Show and tell category)
  - r/LocalLLaMA (mention Gemma fork — different angle, same product)
  - HN: "Show HN: an LLM agent runtime where the rules run every turn"

## What this demonstrates in one shot

1. Mnemara is one `pip install` away.
2. The interesting unit is the **role doc**, not the runtime.
3. The rules are plain Markdown — anyone can edit them.
4. The payoff (halt) is concrete, visible, and saves the viewer money.

That's the hook. Everything else is a consequence.
