**PROBLEM 01**

# Hookline: see how a video will land before it's posted

---

## The Problem

Creators find out whether a video works only after it's posted. By then, the viewers who scrolled past in the first two seconds or dropped off midway are already gone. There is no way today to see, before publishing, how a viewer will experience a video second by second: where they're pulled in, where attention fades, and why.

## What to Build

A system that takes a video, analyses it end to end, and predicts how viewers will respond to it moment by moment. It should tell a creator:

- How strong the hook is, and why
- Where attention is likely to hold and where it is likely to drop
- Which moments are the strongest, and which are the weakest
- What to change, and where in the video, to make it perform better

The analysis should go beyond the script. A video's impact comes from what is seen, heard and said together, and how that changes over time. How you model this is up to you: existing models, your own, or a combination. The output should be something a creator could act on.

> **Judge the video, not the account.** Your system should work from the video's content. Who posted it and how big their page is are only context. At judging, you'll get videos on their own.

## How You'll Be Judged

1. **How you built it:** the architecture, and how a video moves through your system from input to output.
2. **What you considered in the video:** which elements you analyse (visual, audio, speech, pacing, anything else) and why.
3. **Your method:** the models, techniques or logic you used, why you chose them over alternatives, and how the pieces combine into a prediction.
4. **How you know it works:** how you checked your predictions against how real videos actually performed.
5. **Quality of recommendations:** whether your suggestions are specific to moments in the video and would genuinely make it better.

## Data Provided

`videos` `CSV`

About 110 real Instagram reels to get you started: reels from faceless pages (meme, quote and clip pages), reels from face creators, and a small set of extreme viral reels kept separately for reference. Each reel comes with the page's handle, its follower range and the reel's view count. There are no performance labels or watch-time data; gathering reels with known outcomes to test against is part of the task.
