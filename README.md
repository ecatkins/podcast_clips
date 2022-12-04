
## How I thought about the problem
I think there are 3 main things to highlight about the problem.
1. The dataset provided is small
2. There is both audio and text data available in the sample dataset
3. There is no clear objective measure of success


Given #1, I decided to focus on solutions that would not require training/finetuning a model.

With regards to #2, it is likely a customer-facing solution would require analysis of the audio, either to conduct speech-to-text transcription to allow downstream NLP analysis and/or to allow a multi-modal approach that would use both forms of data to clip the audio. However, I decided to focus on the text data, making an assumption that speech-to-text would already be built into the company's existing products.

For #3, the following text was key, that the solution should `highlight engaging or representative spots`. I decided there was no objective way to highlight `engaging` spots in the audio given the data - this would likely require user feedback data, potentially A/B testing different clips of audio, to given a score for measuring engagement. 

`Representative` audio seemed more tractable. Although there is no absolute objective measure I could use, deep learning at it's core is about generating latent representations of something (text/audio/images) - therefore it seemed reasonable to use a model to generate a latent representation of the audio, and then use that representation to find the most similar clip.


## My Solution to the Problem

Given the above, my solution needed to accomplish the following:
1. Generate a representation of the podcast
2. Create candidate clips of audio and generate a representation of each
3. Find the most similar audio clip to the podcast representation

To generate representations of the whole or parts of the podcast, I utilized a pre-trained language model's word embeddings and took the mean of them across any given document. This is a common approach to generate a representation of a document, and is used in many NLP applications. There are definitely more sophisticated approaches to generate these representations, taking into account each word'sc context. But this approach had two main advantages:
1. I did not have to worry about the token limit of the model
2. I could calculate each word's embedding once, and then calculate the "representation" of any sliding window by taking of the mean of the word embeddings in that window - rather than having to run the model on each window.

To generate candidate clips, I again took a simple. Using 150 tokens as a good heuristic of the number of word's typically spoken in a minute, I slid a window over the tokens. For a podcast of N tokens, I would generate N-150 candidate clips.

For each candidate clip, I calculated the representation of the clip, and then calculated the cosine similarity between the podcast representation and the clip representation. The clip with the highest cosine similarity was the most similar clip to the podcast.

I was initially skeptical, that this approach would work well. Intuitively it seemed to be that there was a risk, that this would generate "generic" (boring) parts of the podcast, the opposite of the most engaging parts. But I was pleasantly suprised with the results on the podcasts I tested.

The final part of my solution, was to solve the problem of a clean start to the candidate clip. Given, the sliding window approach, there was no guarantee the audio would start at a natural pause in the podcast. To mitigate this, I searched around the sliding window for the start token, that would maximise the audio break between it, and the previous word. This was a simple heuristic, but it seemed to work well.


## A better approach?

As mentioned above, I think the main risk with my solution, is that it may be attracted to generic pieces of the podcast, rather than the most engaging.

One approach I experimented with, was using a text-summarization model to generate a summary of the podcast. This summary, could then be used to generate a representation of the podcast - this time using a document embedding model. We could then generate candidate clips, also embedding them at the document level, and then finding the most similar clip to the podcast summary.

There were a few obstacles to this approach, that are worth mentioning (and ultimately why I went with the simpler approach above): 
1. Text summarization models are typically trained on news articles, which are typically much shorter than a podcast. I was not sure how well this would work on a podcast. They also typically have a token limit, which would be a problem for a podcast.
2. The sliding window approach to candidate clips, would not work well with a document embedding model. The model would need to be run on each candidate clip, which would be very slow.



















## The AI Solution

## The Deployment Solution









## Things that I excluded from the solution
- Data validation
