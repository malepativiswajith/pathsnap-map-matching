"""Compatibility entry point for the full six-model student-test comparison.

The comparison uses train trips for fitting and the permitted student-test
partition for development metrics. It never reads evaluator-test trips.
"""
from compare_student_models import main


if __name__ == "__main__":
    main()
