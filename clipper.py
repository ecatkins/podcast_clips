import json
import io
import numpy as np


import soundfile as sf
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


class Clipper:
    """Base python class for clipping audio POC"""

    def __init__(self, audio_data, data, save_loc):

        self.audio_data = audio_data
        self.data = data
        self.save_loc = save_loc

    @property
    def item_list(self):
        """Return the list of items from the transcript"""
        return self.data["results"]["items"]

    @property
    def text_list(self):
        """Return the list of text from the transcript"""
        return [item["alternatives"][0]["content"] for item in self.item_list]

    def best_sliding_window(self, word_embeddings, whole_document_embedding, text_list):
        """Find the best sliding window of text based on cosine similarity"""

        # 150 words seems to be a good rule of thumb, for the number of words spoken in 1 minute
        window_size = 150
        max_similarity = 0
        max_similarity_index = 0
        for i in range(len(word_embeddings) - window_size):
            similarity = 1 - cosine(
                whole_document_embedding,
                np.mean(word_embeddings[i : i + window_size], axis=0),
            )
            # print(similarity)
            if similarity > max_similarity:
                max_similarity = similarity
                max_similarity_index = i

        # Print the sentence with the highest similarity and join it
        print(
            " ".join(
                text_list[max_similarity_index : max_similarity_index + window_size]
            )
        )

        window_start = max_similarity_index
        window_end = max_similarity_index + window_size

        return window_start, window_end

    def find_best_start_time(self, window_start, window_end, item_list):
        """Adjust the best start time for the clip based on speech gaps"""

        max_gap = 0
        max_gap_index = 0

        # Look for gaps in speech
        # Start at 50 words before the window start and go 10 words after
        for i in range(window_start - 50, window_start + 10, 1):
            if "start_time" not in item_list[i] or "end_time" not in item_list[i - 1]:
                continue
            diff = item_list[i]["start_time"] - item_list[i - 1]["end_time"]
            if diff > max_gap:
                max_gap = diff
                max_gap_index = i

        print("Max gap: ", max_gap)
        print("Max gap index: ", max_gap_index)

        return max_gap_index

    def cut_audio(self, result):
        """Cut the audio based on the best sliding window using soundfile"""

        item_list = result["item_list"]
        window_start = result["window_start_token"]
        window_end = result["window_end_token"]

        start_time = item_list[window_start]["start_time"]
        end_time = item_list[window_end]["end_time"]

        io_object = io.BytesIO(self.audio_data)
        data, samplerate = sf.read(io_object)
        sf.write(
            self.save_loc,
            data[int(start_time * samplerate) : int(end_time * samplerate)],
            samplerate,
        )

    def run(self):
        """Run the clipper"""

        # Calculate word embeddings for entire entire text list
        print("Calculating embeddings for entire text list")
        text_list = self.text_list
        item_list = self.item_list
        word_embeddings = model.encode(text_list)

        # Calculate whole document average embedding
        print("Calculating average embedding for entire document")
        whole_document_embedding = np.mean(word_embeddings, axis=0)

        # Find best sliding_window
        print("Finding best sliding window")
        window_start, window_end = self.best_sliding_window(
            word_embeddings, whole_document_embedding, text_list
        )
        print(window_start, window_end)

        # Find better start time that aligns with a gap in speech
        print("Find better start time")
        window_start = self.find_best_start_time(window_start, window_end, item_list)

        text = " ".join(
            [
                item["alternatives"][0]["content"]
                for item in item_list[window_start:window_end]
            ]
        )

        result = {
            "window_start_token": window_start,
            "window_end_token": window_end,
            "text": text,
            "item_list": item_list,
        }
        return result
