import re

def estimate_readability(text):
    # Tokenize words
    words = re.findall(r'\b\w+\b', text)
    word_count = len(words)

    # Tokenize sentences (split on punctuation)
    sentences = re.split(r'[.!?]', text)
    sentences_cleaned = [s.strip() for s in sentences if s.strip()]
    sentence_count = len(sentences_cleaned)

    # Characters and syllables
    characters = sum(len(word) for word in words)
    syllable_estimate = sum(len(re.findall(r'[aeiouy]+', word.lower())) for word in words)

    # Calculate readability scores
    flesch_reading_ease = 206.835 - 1.015 * (word_count / sentence_count) - 84.6 * (syllable_estimate / word_count)
    flesch_kincaid_grade = 0.39 * (word_count / sentence_count) + 11.8 * (syllable_estimate / word_count) - 15.59
    average_sentence_length = word_count / sentence_count

    # Output results
    print("Word Count:", word_count)
    print("Sentence Count:", sentence_count)
    print("Average Sentence Length:", round(average_sentence_length, 2))
    print("Estimated Syllables:", syllable_estimate)
    print("Flesch Reading Ease:", round(flesch_reading_ease, 2))
    print("Flesch-Kincaid Grade Level:", round(flesch_kincaid_grade, 2))

if __name__ == "__main__":
    print("Paste your text below. Press Enter twice when you're done:")
    lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        lines.append(line)
    input_text = " ".join(lines)
    estimate_readability(input_text)


def estimate_passive_voice(text):
    sentences = re.split(r'[.!?]', text)
    passive_sentences = 0
    total_sentences = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        total_sentences += 1
        # Look for "to be" verbs followed by past participles
        if re.search(r'\b(is|was|were|are|been|being|be|has been|have been|had been|will be|would be|could be|should be|might be|must be|shall be)\b\s+\w+ed\b', sentence, re.IGNORECASE):
            passive_sentences += 1

    if total_sentences == 0:
        return 0.0
    return round((passive_sentences / total_sentences) * 100, 2)

    # Add passive voice percentage to output
    passive_percentage = estimate_passive_voice(text)
    print("Estimated Passive Sentence %:", passive_percentage)
