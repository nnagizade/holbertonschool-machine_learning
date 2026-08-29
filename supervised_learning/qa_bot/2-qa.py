#!/usr/bin/env python3
"""Module that answers questions from a reference text using a loop,
based on the question_answer function from task 0.
"""
question_answer = __import__('0-qa').question_answer

exit_words = {"exit", "quit", "goodbye", "bye"}


def answer_loop(reference):
    """Answers questions from a reference text in a loop.

    Args:
        reference: the reference text.

    Loops indefinitely, printing 'A: Goodbye' and exiting when the
    user inputs exit/quit/goodbye/bye (case insensitive). If the
    answer cannot be found in the reference text, responds with
    'Sorry, I do not understand your question.'
    """
    while True:
        question = input("Q: ")
        if question.lower() in exit_words:
            print("A: Goodbye")
            break

        answer = question_answer(question, reference)
        if answer is None:
            print("A: Sorry, I do not understand your question.")
        else:
            print("A: {}".format(answer))
