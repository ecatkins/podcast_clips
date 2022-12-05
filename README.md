
# Podcast Clipper 

## The Machine Learning Problem

### How I thought about the problem
I think there are 3 main things to highlight about the problem.
1. The dataset provided is small
2. There is both audio and text data available in the sample dataset
3. There is no clear objective measure of success


Given #1, I decided to focus on solutions that would not require training/finetuning a model.

With regards to #2, it is likely a customer-facing solution would require analysis of the audio, either to conduct speech-to-text transcription to allow downstream NLP analysis and/or to allow a multi-modal approach that would use both forms of data to clip the audio. However, I decided to focus on the text data, making an assumption that speech-to-text would already be built into the company's existing products.

For #3, the following text was key, that the solution should `highlight engaging or representative spots`. I decided there was no effective way to highlight `engaging` spots in the audio given the data - this would likely require user feedback data, potentially A/B testing different clips of audio, to generate a score for measuring engagement. 

`Representative` audio seemed more tractable. Although there is no absolute objective measure I could use, deep learning at it's core is about generating latent representations of something (text/audio/images) - therefore it seemed reasonable to use a model to generate a latent representation of the audio (or the generated text transcript), and use the similarity between the representation and the representation of a candidate clip as a second-order objective measure.


### My Solution to the Problem

Given the above, my solution needed to accomplish the following:
1. Generate a representation of the podcast
2. Create candidate clips of audio and generate a representation of each
3. Find the most similar audio clip to the podcast representation

To generate representations of the whole or parts of the podcast, I utilized a pre-trained language model's word embeddings and took the mean of them across any given document. This is a common approach to generate a representation of a document. There are definitely more sophisticated approaches to generate these representations, one's that take into account the order of the words and the relationships between words. But this approach had two main advantages:
1. I did not have to worry about the token limit of the model
2. I could calculate each word's embedding once, and then calculate the "representation" of any sliding window by taking of the mean of the word embeddings in that window - rather than having to run the model on each window.

To generate candidate clips, I again took a simple approach. Using 150 tokens as a good heuristic of the number of word's typically spoken in a minute, I slid a window over the tokens. For a podcast of N tokens, I would generate N-150 candidate clips.

For each candidate clip, I calculated the representation of the clip, and then calculated the cosine similarity between the podcast representation and the clip representation. The clip with the highest cosine similarity was the most similar clip to the podcast.

I was initially skeptical that this approach would work well. Intuitively it seemed to be that there was a risk that this would generate "generic" (boring) parts of the podcast, the opposite of the most engaging parts. But I was pleasantly suprised with the results on the podcasts I tested.

The final part of my solution, was to solve the problem of a clean start to the candidate clip. Given, the sliding window approach, there was no guarantee the audio would start at a natural pause in the podcast. To mitigate this, I searched around the sliding window for the start token, that would maximise the audio break between it, and the previous word. This was a simple heuristic, but it seemed to work well.


### A better approach?

As mentioned above, I think the main risk with my solution, is that it may be attracted to generic pieces of the podcast, rather than the most engaging.

One approach I experimented with, was using a text-summarization model to generate a summary of the podcast. This summary, could then be used to generate a representation of the podcast - this time using a document embedding model. We could then generate candidate clips, also embedding them at the document level, and then finding the most similar clip to the podcast summary.

There were a couple of obstacles to this approach, that are worth mentioning (and ultimately why I went with the simpler approach above): 
1. Text summarization models are typically trained on news articles, which are typically much shorter than a podcast. I was not sure how well this would work on a podcast. They also have a token limit, which would be a problem for a podcast.
2. The sliding window approach to candidate clips, would not work well with a document embedding model. The model would need to be run on each candidate clip, which would be very slow.

## Implementation & Deployment

Typically for Machine Learning POCs, I find it useful to draw a line between the solution and the deployment. This is because the deployment of a POC, is almost always a temporary implementation. I therefore like to build out a python module that implements the solution, and then I will have the deployment consume that implementation.

This allows me to focus on iterating on the solution (often in a Jupyter Notebook), and then I can focus on the deployment once I am happy with the solution.

NB: I am assuming that the user has access audio transcripts in the same format as the sample data. I have intentionally not built any data validation into this POC, although this is clearly very important for a production solution.

### The Python Module

In this case, I have built the `Clipper` module (see [clipper.py](clipper.py)), which implements the solution. It takes in transcript data, audio data & a desired save location for the audio clip.

It has a two step process for finding the best clip of audio, and then producing the physical audio file based on this.

A more complete run-through is shown in [demo.ipynb](demo.ipynb).

```python
from clipper import Clipper
# Work out the best audio clip
clipper = Clipper(audio_data, data, save_loc='/tmp/clip.wav')
result = clipper.run()
# Cut the given audio file, to produce the slip.
clipper.cut_audio(result)
```

### The API
To minimally deploy the solution, I have utilized FastAPI to build a simple API (see [main.py](main.py)).

It has four main sets of functionality, which are shown in [demo.ipynb](demo.ipynb):
1. Upload a transcript file, which will trigger the solution to find the best clip of audio. It returns an `_id` which is used in the other steps.
2. Upload an audio file (if the user wishes to generate an actual audio file rather than just the clip transcript) using the `_id` returned in Step 1
3. Get the transcript of the best clip of audio using the `_id` returned in Step 1
4. Get the audio file of the best clip of audio using the `_id` returned in Step 1


A full run-through of utilizing the API is shown in [demo.ipynb](demo.ipynb). The demo should allow you to run the API locally, and then interact with it using the `requests` library - at the end of the demo, you should be able to play the resulting audio clip from within the notebook!

One notable decision I made was to utilize a local file system to store the audio and transcript results. This is because I wanted to keep the POC simple, and I did not want to have to worry about setting up a database & large artifact store. However, this is not a scalable solution, and would need to be replaced with a database in a production environment.



