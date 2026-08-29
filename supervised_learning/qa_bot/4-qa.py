#!/usr/bin/env python3
"""Module that answers questions from multiple reference texts."""
semantic_search = __import__('3-semantic_search').semantic_search
question_answer_single = __import__('0-qa').question_answer

exit_words = {"exit", "quit", "goodbye", "bye"}


def question_answer(corpus_path):
    """Answers questions from multiple reference texts.

    Args:
        corpus_path: the path to the corpus of reference documents.

    Loops indefinitely, printing 'A: Goodbye' and exiting when the
    user inputs exit/quit/goodbye/bye (case insensitive). If the
    answer cannot be found, responds with 'Sorry, I do not
    understand your question.'
    """
    while True:
        question = input("Q: ")
        if question.lower() in exit_words:
            print("A: Goodbye")
            break

        reference = semantic_search(corpus_path, question)
        answer = question_answer_single(question, reference)

        if answer is None:
            print("A: Sorry, I do not understand your question.")
        else:
            print("A: {}".format(answer))
