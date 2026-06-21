"""
data/sample_texts.py
--------------------
Sample texts for testing, demos, and experiments.
Each entry has: text, reference (optional), description.
"""

SAMPLES = [
    {
        "id": 1,
        "description": "Subject-verb agreement + homophone errors",
        "text": (
            "The students was very happy and very excited about there new project. "
            "They went to the library and they went to the library again the next day. "
            "The reason why they did this is because of the fact that they needed more books. "
            "It is important to note that the books was old."
        ),
        "reference": (
            "The students were very happy and excited about their new project. "
            "They went to the library twice. "
            "They did this because they needed more books. "
            "The books were old."
        ),
    },
    {
        "id": 2,
        "description": "Double negative + pronoun error + word duplication",
        "text": (
            "She dont know nothing about the the topic. "
            "Me and my friend we both agree that the research is very very important "
            "and the research need more more funding. "
            "Their going to present there findings next week."
        ),
        "reference": (
            "She doesn't know anything about the topic. "
            "My friend and I both agree that the research is very important "
            "and needs more funding. "
            "They are going to present their findings next week."
        ),
    },
    {
        "id": 3,
        "description": "Academic writing — redundancy and wordiness",
        "text": (
            "In order to understand the results of the study, the researchers carefully analysed "
            "the data that was collected. The researchers then presented the data. "
            "The researchers discussed the implications of the data at length. "
            "Due to the fact that the findings were significant, "
            "the researchers decided in order to publish their work. "
            "It is important to note that future plans for additional research have been made."
        ),
        "reference": None,  # No reference — use GBM only
    },
    {
        "id": 4,
        "description": "ESL learner text (CoNLL-2014 style)",
        "text": (
            "With the technology advancing, many people now are able to get informations "
            "from the internet easliy. Although internet give us many convenience, "
            "it also bring many problems for the society. "
            "The reason of this is because peoples spend too much time on internet "
            "and they become very rely on it."
        ),
        "reference": (
            "With technology advancing, many people are now able to get information "
            "from the internet easily. Although the internet gives us many conveniences, "
            "it also brings many problems for society. "
            "This is because people spend too much time on the internet "
            "and have become very reliant on it."
        ),
    },
]


def get_sample(sample_id: int) -> dict:
    """Return a sample by id."""
    for s in SAMPLES:
        if s["id"] == sample_id:
            return s
    raise ValueError(f"Sample {sample_id} not found. Available: {[s['id'] for s in SAMPLES]}")


def list_samples():
    """Print all available samples."""
    for s in SAMPLES:
        ref_label = "has reference" if s["reference"] else "no reference"
        print(f"  [{s['id']}] {s['description']} ({ref_label})")


if __name__ == "__main__":
    print("Available sample texts:\n")
    list_samples()
    print("\nSample 1 text:")
    s = get_sample(1)
    print(s["text"])
