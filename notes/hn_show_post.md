# Hacker News — Show HN draft

## Title (80 char limit)

```
Show HN: Mnemara – an LLM agent runtime where the role doc runs every turn
```

## URL field

```
https://github.com/mekickdemons-creator/mnemara
```

## Text (the comment that opens the thread)

> Mnemara is a small Python runtime around the Claude Agent SDK. The interesting bit
> is that the role doc is re-read on every API call and pinned as `system_prompt`,
> so rules in it apply on turn 20, not just turn 1.
>
> The flagship example is `examples/roles/sentinel.md` — a role doc that makes the
> agent halt on the four common runaway-session failure modes: no-progress, polling,
> drift, and sycophantic reversal. I asked a Claude session running Sentinel to read
> /etc/hostname in a tight loop "until it changes." It refused at turn zero, named
> the failure mode, and offered four bounded alternatives. I pasted the same prompt
> three more times to test repetition; it held.
>
> 18-second demo: https://www.youtube.com/watch?v=mjE0j42l6oU
> (asciinema mirror for terminal purists: https://asciinema.org/a/cId3hYUJrueM9UzU)
>
> Install: `pip install mnemara`. If you have Claude Code installed, no API key
> needed — it uses subscription auth.
>
> MIT licensed. Builds on the Claude Agent SDK. There's a `--set-from-url` command
> that pulls a role doc directly from a raw GitHub URL so you can try Sentinel
> without cloning the repo. Curious whether the "rules every turn" framing works
> for other failure modes people hit.

---

## Posting notes

- HN doesn't render markdown in submission text; the above is plain text already, no formatting needed.
- Submit during US business hours (8am–11am Pacific) for best chance of front-page ranking.
- Do NOT ask people to upvote — flagged immediately as ring-circling.
- If it gets traction, watch for comments and respond fast. The first 30 minutes determine whether it ranks.
- Best replies: concrete, terse, no marketing. If someone asks "isn't this just X" — engage with X, don't deflect.
