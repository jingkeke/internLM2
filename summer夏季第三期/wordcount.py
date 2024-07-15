def wordcount(text):
    """
    Count the occurrences of each word in a given text.

    Args:
    text (str): The input text to count words from.

    Returns:
    dict: A dictionary where keys are words and values are the counts of those words.
    """
    from collections import defaultdict
    import re

    # Use a defaultdict to automatically handle missing keys
    word_counts = defaultdict(int)

    # Use regex to find words, considering words as sequences of alphanumeric characters
    words = re.findall(r'\b\w+\b', text.lower())

    for word in words:
        word_counts[word] += 1

    return dict(word_counts)
if __name__ == "__main__":
    text = "Got this panda plush toy for my daughter's birthday, she loved it! who loves it and takes it everywhere. It's soft and"
    print(wordcount(text))
