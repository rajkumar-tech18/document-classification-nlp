"""
generate_dataset.py
--------------------
Builds a synthetic, offline "news articles by topic" dataset for the
Document Classification project.

Why synthetic data?
Public benchmark corpora such as the 20-Newsgroups dataset require an
internet download at run time. To keep this project fully reproducible
and runnable on any machine (including offline / classroom machines),
this script generates a realistic multi-class news dataset locally.

Each category has several sentence *templates* with slots (e.g. an
actor, a verb, a number) that are filled from category-specific word
lists. Filling templates randomly (instead of drawing from a fixed
list of complete sentences) produces thousands of distinct sentence
combinations, so documents rarely repeat verbatim between the train
and test split -- this keeps the classification task realistically
difficult (~85-95% accuracy) instead of trivially memorizable.

Run:
    python src/generate_dataset.py
Produces:
    data/news_dataset.csv   (columns: text, category)
"""

import csv
import random
from pathlib import Path

random.seed(42)

DOCS_PER_CATEGORY = 220
SENTENCES_PER_DOC = (4, 7)  # inclusive range

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "news_dataset.csv"

# --------------------------------------------------------------------------
# Each category: sentence templates with {slot} placeholders + word lists
# for each slot. Templates deliberately share some generic words (e.g.
# "the", "announced", "this week") across categories, while slot vocabulary
# carries the topic-specific signal -- similar to real news text.
# --------------------------------------------------------------------------

CATEGORIES = {
    "Sports": {
        "team": ["the home side", "the visiting club", "the national squad", "the defending champions",
                 "the underdogs", "the league leaders", "the rookie team"],
        "player": ["the striker", "the captain", "the young midfielder", "the veteran defender",
                   "the star forward", "the goalkeeper", "the new signing"],
        "event": ["the championship final", "the derby match", "the qualifying round",
                  "the season opener", "the playoff series", "the international friendly", "the title decider"],
        "verb_win": ["clinched a dramatic win", "came from behind to triumph", "cruised to victory",
                     "edged out their rivals", "secured a narrow win", "dominated from start to finish"],
        "number": ["two", "three", "four", "a record five", "seven"],
        "templates": [
            "{team} {verb_win} in {event}.",
            "{player} was the standout performer during {event}.",
            "Fans praised {player} after {team} {verb_win}.",
            "{team} will face tough opposition in the next {event}.",
            "The coach credited {player} for scoring {number} goals this season.",
            "Injuries to {player} could affect {team} ahead of {event}.",
            "Analysts expect {team} to reach {event} this year.",
        ],
    },
    "Politics": {
        "official": ["the prime minister", "the finance minister", "the opposition leader",
                     "the newly elected senator", "the state governor", "the foreign secretary", "the president"],
        "body": ["the parliament", "the senate committee", "the cabinet", "the ruling coalition",
                 "the national assembly", "the election commission", "the regional council"],
        "topic": ["the new tax bill", "the healthcare reform", "the trade agreement", "the election law",
                  "the budget proposal", "the constitutional amendment", "the immigration policy"],
        "action": ["debated for hours before approving", "voted narrowly to pass", "postponed a decision on",
                   "held emergency talks over", "released a statement about", "faced criticism over"],
        "number": ["two", "three", "several", "a dozen", "five"],
        "templates": [
            "{body} {action} {topic} this week.",
            "{official} defended {topic} during a press briefing.",
            "Critics say {topic} could face {number} legal challenges.",
            "{official} called on {body} to reconsider {topic}.",
            "Protesters gathered outside {body} to oppose {topic}.",
            "{body} is expected to vote on {topic} next month.",
            "{official} met with {number} foreign delegates to discuss {topic}.",
        ],
    },
    "Technology": {
        "company": ["the startup", "the tech giant", "the chipmaker", "the social media platform",
                    "the cloud provider", "the robotics firm", "the software company"],
        "product": ["a new artificial intelligence model", "a flagship smartphone", "a cloud computing platform",
                    "an autonomous delivery drone", "a cybersecurity tool", "a machine learning framework",
                    "a next-generation processor"],
        "action": ["unveiled", "released a major update to", "began testing", "recalled",
                   "open-sourced", "patched a critical flaw in", "scaled up production of"],
        "metric": ["a million downloads", "record quarterly earnings", "a security breach",
                   "a surge in user growth", "significant delays", "strong developer interest"],
        "number": ["two", "three", "several", "a dozen", "five"],
        "templates": [
            "{company} {action} {product} this week.",
            "{product} from {company} reported {metric}.",
            "Engineers at {company} announced {product} after {number} months of testing.",
            "Analysts say {company} could see {metric} following the release of {product}.",
            "{company} confirmed {metric} linked to {product}.",
            "Developers welcomed {company}'s decision to release {product}.",
            "Investors are watching {company} closely after {metric}.",
        ],
    },
    "Business": {
        "company": ["the retailer", "the manufacturer", "the airline", "the conglomerate",
                    "the energy firm", "the bank", "the logistics company"],
        "metric": ["record quarterly revenue", "a sharp drop in profits", "rising shares", "a wave of layoffs",
                   "strong consumer demand", "a major merger", "an unexpected loss"],
        "cause": ["higher fuel costs", "changing consumer habits", "a new trade agreement",
                  "rising interest rates", "supply chain disruptions", "increased competition", "inflation"],
        "action": ["reported", "announced", "warned investors about", "attributed", "responded to",
                   "revised forecasts because of"],
        "number": ["two", "three", "several", "a dozen", "five"],
        "templates": [
            "{company} {action} {metric} amid {cause}.",
            "Analysts linked {metric} at {company} to {cause}.",
            "{company} announced {metric} in its latest earnings report.",
            "Shares of {company} moved after the firm cited {cause}.",
            "{company} plans to open {number} new locations despite {cause}.",
            "The board of {company} met to discuss {metric}.",
            "Investors reacted cautiously to {cause} affecting {company}.",
        ],
    },
    "Entertainment": {
        "person": ["the actor", "the pop star", "the film director", "the comedian",
                   "the celebrity couple", "the television host", "the fashion designer"],
        "work": ["the new film", "the debut album", "the Broadway musical", "the streaming series",
                 "the world tour", "the documentary", "the awards ceremony"],
        "action": ["received widespread praise for", "was confirmed to headline", "announced plans for",
                   "surprised fans with", "faced criticism over", "celebrated the success of"],
        "metric": ["record ticket sales", "a sold-out premiere", "mixed reviews", "a top chart debut",
                   "strong box office numbers", "a viral social media moment"],
        "number": ["two", "three", "several", "a dozen", "five"],
        "templates": [
            "{person} {action} {work}.",
            "{work} opened to {metric} this weekend.",
            "Fans celebrated as {person} {action} {work}.",
            "Critics highlighted {metric} following the release of {work}.",
            "{person} will appear in {number} new projects after {work}.",
            "{work} broke records with {metric}.",
            "Reports say {person} is finalizing {work} after {metric}.",
        ],
    },
}


# Real news topics overlap in vocabulary (a "tech earnings" story touches both
# Business and Technology, a "player transfer fee" story touches Sports and
# Business, etc). Occasionally borrowing a sentence from a related category
# keeps the task realistically ambiguous instead of trivially separable.
RELATED_CATEGORY = {
    "Business": "Technology",
    "Technology": "Business",
    "Politics": "Business",
    "Sports": "Entertainment",
    "Entertainment": "Sports",
}
CROSSOVER_PROBABILITY = 0.25
LABEL_NOISE_RATE = 0.05  # fraction of documents given a randomly wrong label


def fill_template(template: str, slots: dict) -> str:
    filled = template
    chosen = {}
    for key, options in slots.items():
        if "{" + key + "}" in filled:
            chosen[key] = random.choice(options)
    for key, value in chosen.items():
        filled = filled.replace("{" + key + "}", value)
    return filled[0].upper() + filled[1:]


def random_sentence(category: str) -> str:
    definition = CATEGORIES[category]
    template = random.choice(definition["templates"])
    return fill_template(template, definition)


def build_document(category: str) -> str:
    n_sentences = random.randint(*SENTENCES_PER_DOC)
    sentences = []
    related = RELATED_CATEGORY[category]
    for _ in range(n_sentences):
        if random.random() < CROSSOVER_PROBABILITY:
            sentences.append(random_sentence(related))
        else:
            sentences.append(random_sentence(category))
    return " ".join(sentences)


def main():
    rows = []
    for category in CATEGORIES:
        seen = set()
        attempts = 0
        while len(seen) < DOCS_PER_CATEGORY and attempts < DOCS_PER_CATEGORY * 20:
            doc = build_document(category)
            attempts += 1
            if doc in seen:
                continue
            seen.add(doc)
        for doc in seen:
            rows.append((doc, category))

    # Inject a small amount of label noise to mimic real-world annotation
    # imperfections (e.g. a business/tech crossover story filed under the
    # "wrong" section) rather than a perfectly clean label set.
    all_categories = list(CATEGORIES.keys())
    n_noisy = int(len(rows) * LABEL_NOISE_RATE)
    noisy_indices = random.sample(range(len(rows)), n_noisy)
    for idx in noisy_indices:
        text, true_label = rows[idx]
        wrong_label = random.choice([c for c in all_categories if c != true_label])
        rows[idx] = (text, wrong_label)

    random.shuffle(rows)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "category"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} documents across {len(CATEGORIES)} categories to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
