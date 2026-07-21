#!/usr/bin/env python3
"""Explore country- and user-level topic locality in WildChat-1M."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.compute as pc
from datasets import Dataset, DatasetDict, load_from_disk
from matplotlib import font_manager
from matplotlib.colors import TwoSlopeNorm


DEFAULT_DATASET = Path("/data/huggingface/dataset/wildchat_1m")
DEFAULT_OUTPUT_DIR = Path("artifacts/wildchat_topic_locality")
OTHER = "Other / unclear"
TOPICS = [
    "Code & data",
    "Science & education",
    "Business & work",
    "Health & wellbeing",
    "Society & law",
    "Creative & roleplay",
    "Visual design",
    "Entertainment",
    "Language & writing",
    "Lifestyle & relationships",
]
TOPIC_SHORT_NAMES = {
    "Code & data": "Code &\ndata",
    "Science & education": "Science &\neducation",
    "Business & work": "Business &\nwork",
    "Health & wellbeing": "Health &\nwellbeing",
    "Society & law": "Society &\nlaw",
    "Creative & roleplay": "Creative &\nroleplay",
    "Visual design": "Visual\ndesign",
    "Entertainment": "Entertain-\nment",
    "Language & writing": "Language &\nwriting",
    "Lifestyle & relationships": "Lifestyle &\nrelationships",
}


def _patterns(*items: Tuple[int, str]) -> List[Tuple[int, re.Pattern[str]]]:
    return [(weight, re.compile(pattern, re.IGNORECASE | re.DOTALL)) for weight, pattern in items]


TOPIC_PATTERNS: Dict[str, List[Tuple[int, re.Pattern[str]]]] = {
    "Code & data": _patterns(
        (4, r"```(?:python|java|javascript|typescript|c\+\+|c#|sql|html|css|bash|r\b|go\b|rust)"),
        (4, r"\b(?:write|fix|debug|review|refactor|generate).{0,45}\b(?:code|program|script|query|function)\b"),
        (3, r"\b(?:python|javascript|typescript|java|c\+\+|c#|php|golang|rust|swift|kotlin|matlab|vba)\b"),
        (3, r"\b(?:sql|database|api|json|regex|docker|kubernetes|linux|github|git\b|excel formula|power ?bi)\b"),
        (2, r"\b(?:programming|software|web development|data analysis|machine learning|algorithm|source code|compiler)\b"),
        (2, r"\b(?:bug|stack trace|exception|syntax error|runtime error)\b"),
    ),
    "Science & education": _patterns(
        (4, r"\b(?:solve|calculate|derive|prove).{0,45}\b(?:equation|integral|derivative|theorem|math problem|probability)\b"),
        (3, r"\b(?:calculus|algebra|geometry|statistics|physics|chemistry|biology|astronomy|geology)\b"),
        (3, r"\b(?:homework|assignment|lesson plan|quiz|exam question|thesis|dissertation)\b"),
        (2, r"\b(?:scientific|academic|research paper|experiment|student|teacher|university|study guide)\b"),
        (2, r"\b(?:equation|molecule|quantum|evolution|ecosystem|cell biology)\b"),
    ),
    "Business & work": _patterns(
        (4, r"\b(?:seo|search engine optimization|business plan|marketing plan|market research|sales strategy)\b"),
        (3, r"\b(?:business|company|startup|entrepreneur|customer|marketing|advertising|sales|e-?commerce)\b"),
        (3, r"\b(?:finance|financial|investment|investing|stock market|accounting|revenue|profit|budget)\b"),
        (3, r"\b(?:resume|curriculum vitae|cover letter|job interview|project management|product manager)\b"),
        (2, r"\b(?:professional email|proposal|workplace|career|client|brand|linkedin)\b"),
    ),
    "Health & wellbeing": _patterns(
        (4, r"\b(?:diagnos|symptom|treatment|medication|medicine|doctor|medical advice)\w*\b"),
        (3, r"\b(?:health|disease|infection|cancer|diabetes|blood pressure|pregnan|surgery|hospital)\w*\b"),
        (3, r"\b(?:mental health|depression|anxiety|therapy|therapist|adhd|autism)\b"),
        (2, r"\b(?:nutrition|diet|calories|workout|exercise|fitness|sleep|wellbeing|well-being)\b"),
        (2, r"\b(?:headache|stomach|chest pain|skin rash|vitamin|protein intake)\b"),
    ),
    "Society & law": _patterns(
        (4, r"\b(?:legal advice|constitutional law|criminal law|international law|human rights)\b"),
        (3, r"\b(?:politic|government|election|president|parliament|congress|public policy|geopolitic)\w*\b"),
        (3, r"\b(?:law|legal|court|lawsuit|crime|police|rights|regulation|immigration)\b"),
        (3, r"\b(?:history|historical|war|religion|ethnic|racism|society|social movement)\b"),
        (2, r"\b(?:democracy|communism|capitalism|country|nation|culture|philosophy|ethics)\b"),
    ),
    "Creative & roleplay": _patterns(
        (5, r"\b(?:write|create|continue|tell).{0,35}\b(?:story|fiction|fanfic|poem|novel|screenplay|dialogue|scene)\b"),
        (4, r"\b(?:roleplay|role-play|role play|fictional story|creative writing|character profile)\b"),
        (3, r"\b(?:vividly detailed|comedic story|fantasy world|in the clubroom|there would be dialogue)\b"),
        (2, r"\b(?:plot|protagonist|villain|character|narrative|chapter|poetry)\b"),
    ),
    "Visual design": _patterns(
        (5, r"\b(?:midjourney|stable diffusion|dall-?e|image-generation prompt|text-to-image)\b"),
        (5, r"\b(?:generate|create|write).{0,35}\b(?:image prompt|prompt for an image|logo|illustration|poster|thumbnail)\b"),
        (3, r"\b(?:graphic design|visual design|digital art|photography|artwork|image prompt)\b"),
        (2, r"\b(?:camera angle|aspect ratio|photorealistic|cinematic lighting|watercolor|oil painting)\b"),
    ),
    "Entertainment": _patterns(
        (4, r"\b(?:movie|film|tv show|television series|video game|anime|manga)\b"),
        (3, r"\b(?:celebrity|actor|actress|musician|singer|band|album|song|youtube|streamer)\b"),
        (3, r"\b(?:football|basketball|baseball|soccer|tennis|wrestling|sports)\b"),
        (2, r"\b(?:star wars|marvel|dc comics|naruto|pokemon|minecraft|cast|episode)\b"),
    ),
    "Language & writing": _patterns(
        (5, r"\b(?:translate|translation).{0,25}\b(?:english|french|spanish|german|chinese|arabic|language|into|to)\b"),
        (4, r"\b(?:proofread|paraphrase|rewrite|rephrase|summarize|summarise|grammar check)\b"),
        (4, r"\b(?:write|draft|compose).{0,35}\b(?:article|essay|paragraph|email|letter|response|description|caption)\b"),
        (3, r"\b(?:grammar|vocabulary|language learning|word meaning|pronunciation)\b"),
        (2, r"\b(?:tone|spelling|concise|word count|bullet points)\b"),
    ),
    "Lifestyle & relationships": _patterns(
        (4, r"\b(?:relationship advice|dating advice|travel itinerary|meal plan|recipe for)\b"),
        (3, r"\b(?:relationship|boyfriend|girlfriend|husband|wife|marriage|dating|breakup|family|parenting)\b"),
        (3, r"\b(?:travel|trip|vacation|hotel|restaurant|recipe|cooking|fashion|wedding|home decor)\b"),
        (2, r"\b(?:friendship|personal advice|recommendation|hobby|pet|garden|car|shopping)\b"),
    ),
}

# Cheap substring screening avoids running every regular expression on every
# request. Each tuple intentionally contains all concepts used by that topic's
# higher-precision patterns above.
TOPIC_HINTS: Dict[str, Tuple[str, ...]] = {
    "Code & data": (
        "code", "program", "script", "python", "java", "sql", "database", "api", "json",
        "regex", "docker", "kubernetes", "linux", "github", "git ", "excel", "power bi",
        "software", "algorithm", "compiler", "bug", "exception", "error", "machine learning",
        "data analysis", "html", "css", "php", "golang", "rust", "swift", "kotlin", "matlab",
    ),
    "Science & education": (
        "solve", "calculate", "derive", "prove", "equation", "integral", "derivative", "theorem",
        "math", "probability", "calculus", "algebra", "geometry", "statistics", "physics",
        "chemistry", "biology", "astronomy", "geology", "homework", "assignment", "lesson plan",
        "quiz", "exam", "thesis", "dissertation", "scientific", "academic", "research paper",
        "experiment", "student", "teacher", "university", "study guide", "molecule", "quantum",
        "evolution", "ecosystem",
    ),
    "Business & work": (
        "seo", "business", "marketing", "market research", "sales", "company", "startup",
        "entrepreneur", "customer", "advertising", "commerce", "finance", "financial", "investment",
        "stock market", "accounting", "revenue", "profit", "budget", "resume", "curriculum vitae",
        "cover letter", "job interview", "project management", "product manager", "professional email",
        "proposal", "workplace", "career", "client", "brand", "linkedin",
    ),
    "Health & wellbeing": (
        "diagnos", "symptom", "treatment", "medication", "medicine", "doctor", "medical", "health",
        "disease", "infection", "cancer", "diabetes", "blood pressure", "pregnan", "surgery",
        "hospital", "mental health", "depression", "anxiety", "therapy", "therapist", "adhd", "autism",
        "nutrition", "diet", "calories", "workout", "exercise", "fitness", "sleep", "wellbeing",
        "well-being", "headache", "stomach", "chest pain", "skin rash", "vitamin", "protein intake",
    ),
    "Society & law": (
        "legal", "law", "constitutional", "criminal", "human rights", "politic", "government",
        "election", "president", "parliament", "congress", "public policy", "geopolitic", "court",
        "lawsuit", "crime", "police", "rights", "regulation", "immigration", "history", "historical",
        "war", "religion", "ethnic", "racism", "society", "social movement", "democracy", "communism",
        "capitalism", "country", "nation", "culture", "philosophy", "ethics",
    ),
    "Creative & roleplay": (
        "write", "create", "continue", "tell", "story", "fiction", "fanfic", "poem", "novel",
        "screenplay", "dialogue", "scene", "roleplay", "role-play", "role play", "creative writing",
        "character", "vividly detailed", "comedic story", "fantasy world", "clubroom", "plot",
        "protagonist", "villain", "narrative", "chapter", "poetry",
    ),
    "Visual design": (
        "midjourney", "stable diffusion", "dall-e", "dalle", "image prompt", "prompt for an image",
        "text-to-image", "logo", "illustration", "poster", "thumbnail", "graphic design", "visual design",
        "digital art", "photography", "artwork", "camera angle", "aspect ratio", "photorealistic",
        "cinematic lighting", "watercolor", "oil painting",
    ),
    "Entertainment": (
        "movie", "film", "tv show", "television", "video game", "anime", "manga", "celebrity", "actor",
        "actress", "musician", "singer", "band", "album", "song", "youtube", "streamer", "football",
        "basketball", "baseball", "soccer", "tennis", "wrestling", "sports", "star wars", "marvel",
        "dc comics", "naruto", "pokemon", "minecraft", "cast", "episode",
    ),
    "Language & writing": (
        "translate", "translation", "english", "french", "spanish", "german", "chinese", "arabic",
        "language", "proofread", "paraphrase", "rewrite", "rephrase", "summarize", "summarise",
        "grammar", "write", "draft", "compose", "article", "essay", "paragraph", "email", "letter",
        "response", "description", "caption", "vocabulary", "word meaning", "pronunciation", "tone",
        "spelling", "concise", "word count", "bullet points",
    ),
    "Lifestyle & relationships": (
        "relationship", "dating", "travel", "itinerary", "meal plan", "recipe", "boyfriend", "girlfriend",
        "husband", "wife", "marriage", "breakup", "family", "parenting", "trip", "vacation", "hotel",
        "restaurant", "cooking", "fashion", "wedding", "home decor", "friendship", "personal advice",
        "recommendation", "hobby", "pet", "garden", "car", "shopping",
    ),
}

PRIMARY_INTENT_PATTERNS: List[Tuple[str, re.Pattern[str]]] = [
    (
        "Visual design",
        re.compile(
            r"\b(?:midjourney|stable diffusion|dall-?e|text-to-image|image-generation prompt)\b|"
            r"\b(?:generate|create|write).{0,35}\b(?:image prompt|prompt for an image|logo|illustration|poster|thumbnail)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "Code & data",
        re.compile(
            r"```(?:python|java|javascript|typescript|c\+\+|c#|sql|html|css|bash|r\b|go\b|rust)|"
            r"\b(?:write|fix|debug|review|refactor|generate).{0,45}\b(?:code|program|query|function)\b|"
            r"\b(?:write|fix|debug|review).{0,35}\b(?:python|bash|shell|lua|powershell|javascript) script\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "Language & writing",
        re.compile(
            r"\b(?:translate|translation)\b|"
            r"\b(?:proofread|paraphrase|rewrite|rephrase|summarize|summarise|grammar check)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "Creative & roleplay",
        re.compile(
            r"\b(?:write|create|continue|tell|make|describe|detail).{0,35}\b(?:story|fiction|fanfic|poem|novel|screenplay|dialogue|scene)\b|"
            r"\b(?:roleplay|role-play|role play|fictional story|creative writing)\b|"
            r"\b(?:fantasy|fictional|dystopian).{0,25}\b(?:story|world|universe|setting)\b|"
            r"\bsetting up.{0,30}\b(?:story|world|universe)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
]


def _configure_font() -> None:
    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        family = font_manager.FontProperties(fname=str(font_path)).get_name()
    else:
        family = "DejaVu Sans"
    plt.rcParams.update(
        {
            "font.family": family,
            "axes.unicode_minus": False,
            "font.size": 9.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def classify_topic(text: str) -> Tuple[str, int]:
    """Return a conservative, auditable topic label and its rule score."""
    normalized = " ".join(str(text).lower().split())[:3000]
    for topic, pattern in PRIMARY_INTENT_PATTERNS:
        if pattern.search(normalized):
            return topic, 10
    scores = {
        topic: (
            sum(weight for weight, pattern in patterns if pattern.search(normalized))
            if any(hint in normalized for hint in TOPIC_HINTS[topic])
            else 0
        )
        for topic, patterns in TOPIC_PATTERNS.items()
    }
    best_score = max(scores.values(), default=0)
    if best_score < 4:
        return OTHER, best_score
    # TOPICS defines deterministic tie-breaking, favoring specific service domains.
    best_topic = next(topic for topic in TOPICS if scores[topic] == best_score)
    return best_topic, best_score


def _first_user_request(conversation: Sequence[Mapping[str, Any]]) -> str:
    for turn in conversation:
        if turn.get("role") == "user" and turn.get("content"):
            return str(turn["content"])
    return ""


def _load_dataset(path: Path) -> Dataset:
    loaded = load_from_disk(str(path))
    if isinstance(loaded, DatasetDict):
        return loaded["train"]
    return loaded


def _collect_counts(
    dataset: Dataset,
    batch_size: int,
) -> Tuple[
    Counter[str],
    Dict[str, Counter[str]],
    Dict[str, Counter[str]],
    Dict[Tuple[str, str], Counter[str]],
    Dict[str, Any],
    Dict[str, List[str]],
]:
    table = dataset.data.table.select(["language", "country", "hashed_ip", "conversation"])
    global_topics: Counter[str] = Counter()
    country_topics: Dict[str, Counter[str]] = defaultdict(Counter)
    user_topics: Dict[str, Counter[str]] = defaultdict(Counter)
    country_user_topics: Dict[Tuple[str, str], Counter[str]] = defaultdict(Counter)
    examples: Dict[str, List[str]] = defaultdict(list)
    stats: Counter[str] = Counter()
    stats["all_conversations"] = len(dataset)
    for record_batch in table.to_batches(max_chunksize=batch_size):
        english = pc.equal(record_batch.column("language"), "English")
        filtered = record_batch.filter(english)
        if filtered.num_rows == 0:
            continue
        first_turn = pc.list_element(filtered.column("conversation"), 0)
        batch = {
            "country": filtered.column("country").to_pylist(),
            "hashed_ip": filtered.column("hashed_ip").to_pylist(),
            "role": pc.struct_field(first_turn, "role").to_pylist(),
            "content": pc.struct_field(first_turn, "content").to_pylist(),
        }
        stats["english_conversations"] += filtered.num_rows
        rows = zip(
            batch["country"],
            batch["hashed_ip"],
            batch["role"],
            batch["content"],
        )
        for country, hashed_ip, role, text in rows:
            if role != "user" or not text:
                stats["empty_requests"] += 1
                continue
            topic, score = classify_topic(str(text))
            stats["topic_" + topic] += 1
            if topic == OTHER:
                continue
            stats["classified_requests"] += 1
            global_topics[topic] += 1
            if country:
                country_topics[str(country)][topic] += 1
                stats["geolocated_classified_requests"] += 1
            if hashed_ip:
                user_topics[str(hashed_ip)][topic] += 1
                stats["identified_classified_requests"] += 1
                if country:
                    country_user_topics[(str(country), str(hashed_ip))][topic] += 1
            if len(examples[topic]) < 12 and score >= 4:
                examples[topic].append(" ".join(text.split())[:500])
    return global_topics, country_topics, user_topics, country_user_topics, dict(stats), examples


def _cap_country_user_contributions(
    country_user_topics: Mapping[Tuple[str, str], Counter[str]],
    max_requests_per_user: int,
) -> Tuple[Dict[str, Counter[str]], Counter[str]]:
    """Limit each hashed IP's influence on its country's topic distribution."""
    capped: Dict[str, Counter[str]] = defaultdict(Counter)
    global_capped: Counter[str] = Counter()
    for (country, _), counts in country_user_topics.items():
        total = sum(counts.values())
        weight = min(1.0, max_requests_per_user / total)
        for topic, count in counts.items():
            contribution = count * weight
            capped[country][topic] += contribution
            global_capped[topic] += contribution
    return capped, global_capped


def _entropy(counts: Mapping[str, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return -sum(
        (count / total) * math.log(count / total)
        for count in counts.values()
        if count > 0
    )


def _group_metrics(
    groups: Mapping[str, Counter[str]],
    min_requests: int,
) -> List[Dict[str, Any]]:
    rows = []
    for group, counts in groups.items():
        total = sum(counts.values())
        if total < min_requests:
            continue
        dominant_topic, dominant_count = counts.most_common(1)[0]
        rows.append(
            {
                "group": group,
                "requests": total,
                "dominant_topic": dominant_topic,
                "dominant_share": dominant_count / total,
                "normalized_entropy": _entropy(counts) / math.log(len(TOPICS)),
            }
        )
    return rows


def _explained_entropy(
    global_counts: Mapping[str, int],
    group_counts: Mapping[str, Counter[str]],
    eligible_groups: Iterable[str],
) -> float:
    eligible = set(eligible_groups)
    joint_total = sum(sum(group_counts[group].values()) for group in eligible)
    if joint_total <= 0:
        return 0.0
    topic_counts = Counter()
    for group in eligible:
        topic_counts.update(group_counts[group])
    topic_entropy = _entropy(topic_counts)
    conditional_entropy = sum(
        (sum(group_counts[group].values()) / joint_total) * _entropy(group_counts[group])
        for group in eligible
    )
    del global_counts
    return (topic_entropy - conditional_entropy) / topic_entropy if topic_entropy else 0.0


def _multinomial_baseline(
    group_rows: Sequence[Mapping[str, Any]],
    global_counts: Mapping[str, int],
    repetitions: int,
    seed: int,
) -> np.ndarray:
    probabilities = np.asarray([global_counts[topic] for topic in TOPICS], dtype=float)
    probabilities /= probabilities.sum()
    sizes = np.asarray([row["requests"] for row in group_rows], dtype=int)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repetitions):
        for size in sizes:
            sample = rng.multinomial(int(size), probabilities)
            values.append(float(sample.max() / size))
    return np.asarray(values)


def _jensen_shannon_divergence(
    counts: Mapping[str, float],
    reference_counts: Mapping[str, float],
) -> float:
    values = np.asarray([counts[topic] for topic in TOPICS], dtype=float)
    reference = np.asarray([reference_counts[topic] for topic in TOPICS], dtype=float)
    values /= values.sum()
    reference /= reference.sum()
    midpoint = (values + reference) / 2.0

    def kl_divergence(left: np.ndarray, right: np.ndarray) -> float:
        mask = left > 0
        return float(np.sum(left[mask] * np.log2(left[mask] / right[mask])))

    return 0.5 * kl_divergence(values, midpoint) + 0.5 * kl_divergence(reference, midpoint)


def _divergence_baseline(
    group_rows: Sequence[Mapping[str, Any]],
    global_counts: Mapping[str, int],
    repetitions: int,
    seed: int,
) -> np.ndarray:
    probabilities = np.asarray([global_counts[topic] for topic in TOPICS], dtype=float)
    probabilities /= probabilities.sum()
    rng = np.random.default_rng(seed)
    values = []
    reference = {topic: global_counts[topic] for topic in TOPICS}
    for _ in range(repetitions):
        for row in group_rows:
            sample = rng.multinomial(int(row["requests"]), probabilities)
            counts = {topic: sample[index] for index, topic in enumerate(TOPICS)}
            values.append(_jensen_shannon_divergence(counts, reference))
    return np.asarray(values)


def _heatmap(
    groups: Sequence[str],
    group_counts: Mapping[str, Counter[str]],
    global_counts: Mapping[str, int],
    mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    shares = np.zeros((len(groups), len(TOPICS)), dtype=float)
    colors = np.zeros_like(shares)
    global_total = sum(global_counts.values())
    for row, group in enumerate(groups):
        total = sum(group_counts[group].values())
        for column, topic in enumerate(TOPICS):
            count = group_counts[group][topic]
            shares[row, column] = count / total
            if mode == "lift":
                local = (count + 0.5) / (total + 0.5)
                global_share = global_counts[topic] / global_total
                colors[row, column] = math.log2(local / global_share)
            else:
                colors[row, column] = shares[row, column]
    return shares, colors


def _render_heatmaps(
    output_stem: Path,
    country_groups: Sequence[str],
    country_counts: Mapping[str, Counter[str]],
    user_groups: Sequence[str],
    user_counts: Mapping[str, Counter[str]],
    country_global_counts: Mapping[str, int],
    user_global_counts: Mapping[str, int],
    classified_requests: int,
    user_aliases: Mapping[str, str] | None = None,
) -> Dict[str, str]:
    _configure_font()
    country_shares, country_lifts = _heatmap(
        country_groups, country_counts, country_global_counts, "lift"
    )
    user_shares, _ = _heatmap(user_groups, user_counts, user_global_counts, "share")
    fig, (ax_country, ax_user) = plt.subplots(2, 1, figsize=(13.2, 10.4))

    country_image = ax_country.imshow(
        np.clip(country_lifts, -2.5, 2.5),
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-2.5, vcenter=0.0, vmax=2.5),
        aspect="auto",
    )
    ax_country.set_yticks(np.arange(len(country_groups)), country_groups)
    ax_country.set_xticks(np.arange(len(TOPICS)), [TOPIC_SHORT_NAMES[t] for t in TOPICS])
    ax_country.tick_params(length=0)
    ax_country.spines[:].set_visible(False)
    ax_country.set_title(
        "A  Countries have distinct topic mixes",
        loc="left",
        fontsize=13,
        weight="bold",
        pad=10,
    )
    for row in range(len(country_groups)):
        for column in range(len(TOPICS)):
            share = country_shares[row, column] * 100.0
            if share < 3.0:
                continue
            color = "white" if abs(country_lifts[row, column]) >= 1.5 else "#202020"
            ax_country.text(column, row, f"{share:.0f}", ha="center", va="center", fontsize=7.4, color=color)
    cbar = fig.colorbar(country_image, ax=ax_country, fraction=0.019, pad=0.02)
    cbar.set_label("Demand lift vs. global topic mix (log2)")
    cbar.set_ticks([-2, -1, 0, 1, 2], labels=["1/4x", "1/2x", "1x", "2x", "4x"])

    user_image = ax_user.imshow(user_shares, cmap="YlOrRd", vmin=0.0, vmax=1.0, aspect="auto")
    user_labels = [
        f"{user_aliases.get(group, f'User {index + 1:02d}') if user_aliases else f'User {index + 1:02d}'}  "
        f"(n={sum(user_counts[group].values()):.0f})"
        for index, group in enumerate(user_groups)
    ]
    ax_user.set_yticks(np.arange(len(user_groups)), user_labels)
    ax_user.set_xticks(np.arange(len(TOPICS)), [TOPIC_SHORT_NAMES[t] for t in TOPICS])
    ax_user.tick_params(length=0)
    ax_user.spines[:].set_visible(False)
    ax_user.set_title(
        "B  Active users concentrate on different topics",
        loc="left",
        fontsize=13,
        weight="bold",
        pad=10,
    )
    for row in range(len(user_groups)):
        for column in range(len(TOPICS)):
            share = user_shares[row, column] * 100.0
            if share < 6.0:
                continue
            color = "white" if share >= 52.0 else "#202020"
            ax_user.text(column, row, f"{share:.0f}", ha="center", va="center", fontsize=7.4, color=color)
    cbar = fig.colorbar(user_image, ax=ax_user, fraction=0.019, pad=0.02)
    cbar.set_label("Share of user's classified requests")
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0], labels=["0%", "25%", "50%", "75%", "100%"])

    fig.suptitle("WildChat topics exhibit geographic and user locality", fontsize=16, weight="bold", y=0.995)
    fig.text(
        0.5,
        0.012,
        f"English first-turn requests; {classified_requests:,} conservatively classified by auditable intent rules. "
        "Cell labels are within-group shares (%).",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.965), h_pad=2.8)
    png = output_stem.with_suffix(".png")
    pdf = output_stem.with_suffix(".pdf")
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png), "pdf": str(pdf)}


def _render_concentration(
    output_stem: Path,
    country_divergence: np.ndarray,
    user_rows: Sequence[Mapping[str, Any]],
    country_divergence_baseline: np.ndarray,
    user_baseline: np.ndarray,
    country_explained: float,
    user_explained: float,
) -> Dict[str, str]:
    _configure_font()
    orange = "#d46b45"
    blue = "#2b6f9f"
    fig, (ax_country, ax_user) = plt.subplots(1, 2, figsize=(11.4, 4.7))

    def plot_cdf(
        ax: Any,
        actual: np.ndarray,
        baseline: np.ndarray,
        title: str,
        xlabel: str,
        scale: float = 100.0,
    ) -> None:
        for values, label, color, linestyle in (
            (actual, "Observed", orange, "-"),
            (baseline, "Shuffled topic mix", blue, "--"),
        ):
            sorted_values = np.sort(values) * scale
            y = np.arange(1, len(sorted_values) + 1) / len(sorted_values) * 100.0
            ax.plot(sorted_values, y, color=color, linestyle=linestyle, linewidth=2.5, label=label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Groups at or below x (%)")
        ax.set_title(title, loc="left", fontsize=13, weight="bold", pad=10)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, loc="lower right")

    user_actual = np.asarray([row["dominant_share"] for row in user_rows])
    plot_cdf(
        ax_country,
        country_divergence,
        country_divergence_baseline,
        "A  Country topic-mix divergence",
        "Divergence from global topic mix (bits)",
        scale=1.0,
    )
    plot_cdf(
        ax_user,
        user_actual,
        user_baseline,
        "B  User-level concentration",
        "Dominant topic share within user (%)",
    )
    ax_country.text(
        0.03,
        0.94,
        f"Country explains {country_explained * 100:.1f}%\nof topic entropy",
        transform=ax_country.transAxes,
        va="top",
        fontsize=9,
        color="#444444",
    )
    ax_user.text(
        0.03,
        0.94,
        f"User explains {user_explained * 100:.1f}%\nof topic entropy",
        transform=ax_user.transAxes,
        va="top",
        fontsize=9,
        color="#444444",
    )
    fig.suptitle("Topic locality appears at country and user scopes", fontsize=15, weight="bold", y=0.995)
    fig.text(
        0.5,
        0.015,
        "Countries are capped at 20 request-equivalents per hashed IP; eligible users have >=10 requests. "
        "Null preserves group sizes and the global topic mix.",
        ha="center",
        fontsize=8.8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95), w_pad=3.0)
    png = output_stem.with_suffix(".png")
    pdf = output_stem.with_suffix(".pdf")
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png), "pdf": str(pdf)}


def _null_metric_means(
    group_rows: Sequence[Mapping[str, Any]],
    global_counts: Mapping[str, int],
    repetitions: int,
    seed: int,
    metric: str,
) -> np.ndarray:
    probabilities = np.asarray([global_counts[topic] for topic in TOPICS], dtype=float)
    probabilities /= probabilities.sum()
    reference = {topic: global_counts[topic] for topic in TOPICS}
    rng = np.random.default_rng(seed)
    means = []
    for row in group_rows:
        size = max(1, int(row["requests"]))
        samples = rng.multinomial(size, probabilities, size=repetitions)
        if metric == "dominant_share":
            values = samples.max(axis=1) / size
        elif metric == "divergence":
            values = np.asarray(
                [
                    _jensen_shannon_divergence(
                        {topic: sample[index] for index, topic in enumerate(TOPICS)},
                        reference,
                    )
                    for sample in samples
                ]
            )
        else:
            raise ValueError(f"unsupported null metric: {metric}")
        means.append(float(np.mean(values)))
    return np.asarray(means)


def _render_scatter(
    output_stem: Path,
    country_rows: Sequence[Mapping[str, Any]],
    country_counts: Mapping[str, Counter[str]],
    country_global_counts: Mapping[str, int],
    user_rows: Sequence[Mapping[str, Any]],
    user_global_counts: Mapping[str, int],
    repetitions: int,
    seed: int,
) -> Dict[str, str]:
    """Render concrete group-level observations against volume-matched nulls."""
    _configure_font()
    orange = "#d46b45"
    blue = "#2b6f9f"
    country_sizes = np.asarray([float(row["requests"]) for row in country_rows])
    country_actual = np.asarray(
        [
            _jensen_shannon_divergence(
                country_counts[str(row["group"])], country_global_counts
            )
            for row in country_rows
        ]
    )
    country_null = _null_metric_means(
        country_rows,
        country_global_counts,
        repetitions,
        seed,
        "divergence",
    )
    user_sizes = np.asarray([float(row["requests"]) for row in user_rows])
    user_actual = np.asarray([float(row["dominant_share"]) for row in user_rows]) * 100.0
    user_null = (
        _null_metric_means(
            user_rows,
            user_global_counts,
            repetitions,
            seed + 1,
            "dominant_share",
        )
        * 100.0
    )

    fig, (ax_country, ax_user) = plt.subplots(1, 2, figsize=(12.0, 5.0))
    for x, low, high in zip(country_sizes, country_null, country_actual):
        ax_country.plot([x, x], [low, high], color="#c8cdd2", linewidth=0.9, zorder=1)
    ax_country.scatter(
        country_sizes,
        country_null,
        s=34,
        marker="x",
        linewidth=1.5,
        color=blue,
        label="Volume-matched null",
        zorder=2,
    )
    ax_country.scatter(
        country_sizes,
        country_actual,
        s=42,
        color=orange,
        edgecolor="white",
        linewidth=0.6,
        label="Observed country",
        zorder=3,
    )
    label_indices = sorted(
        set(np.argsort(country_sizes)[-6:].tolist() + np.argsort(country_actual)[-3:].tolist())
    )
    label_offsets = {"Singapore": (4, -12), "Japan": (4, 5)}
    for index in label_indices:
        country = str(country_rows[index]["group"])
        ax_country.annotate(
            country,
            (country_sizes[index], country_actual[index]),
            xytext=label_offsets.get(country, (4, 4)),
            textcoords="offset points",
            fontsize=7.6,
            color="#404040",
        )
    ax_country.set_xscale("log")
    ax_country.set_xlabel("Classified request-equivalents in country (log scale)")
    ax_country.set_ylabel("Divergence from global topic mix (bits)")
    ax_country.set_title("A  Each country vs. its random baseline", loc="left", fontsize=13, weight="bold", pad=10)
    ax_country.grid(alpha=0.2)
    ax_country.spines[["top", "right"]].set_visible(False)
    ax_country.legend(frameon=False, loc="upper right")
    ax_country.text(
        0.98,
        0.79,
        f"Median: {np.median(country_actual):.3f} bits observed\n"
        f"vs. {np.median(country_null):.4f} bits under null",
        transform=ax_country.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        color="#555555",
    )

    ax_user.scatter(
        user_sizes,
        user_null,
        s=11,
        color=blue,
        alpha=0.25,
        edgecolors="none",
        label="Volume-matched null",
        rasterized=True,
    )
    ax_user.scatter(
        user_sizes,
        user_actual,
        s=13,
        color=orange,
        alpha=0.30,
        edgecolors="none",
        label="Observed user",
        rasterized=True,
    )
    ax_user.set_xscale("log")
    ax_user.set_ylim(0, 103)
    ax_user.set_xlabel("Classified requests per hashed-IP user (log scale)")
    ax_user.set_ylabel("Dominant topic share within user (%)")
    ax_user.set_title("B  Each user vs. its random baseline", loc="left", fontsize=13, weight="bold", pad=10)
    ax_user.grid(alpha=0.2)
    ax_user.spines[["top", "right"]].set_visible(False)
    ax_user.legend(frameon=False, loc="lower right")
    ax_user.text(
        0.03,
        0.04,
        f"Median: {np.median(user_actual):.0f}% observed\n"
        f"vs. {np.median(user_null):.1f}% under null",
        transform=ax_user.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        color="#555555",
    )

    fig.suptitle("Observed topic locality is visible group by group", fontsize=15, weight="bold", y=0.995)
    fig.text(
        0.5,
        0.014,
        "Countries are capped at 20 request-equivalents per hashed IP; users have >=10 classified requests. "
        "Blue points are per-group means over shuffled assignments.",
        ha="center",
        fontsize=8.8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95), w_pad=3.2)
    png = output_stem.with_suffix(".png")
    pdf = output_stem.with_suffix(".pdf")
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png), "pdf": str(pdf)}


def _write_group_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    group_counts: Mapping[str, Counter[str]],
    aliases: Mapping[str, str] | None = None,
) -> None:
    fields = ["group", "requests", "dominant_topic", "dominant_share", "normalized_entropy"] + TOPICS
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            group = str(row["group"])
            output = dict(row)
            output["group"] = aliases.get(group, group) if aliases else group
            output.update({topic: group_counts[group][topic] for topic in TOPICS})
            writer.writerow(output)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-countries", type=int, default=16)
    parser.add_argument("--num-users", type=int, default=18)
    parser.add_argument("--min-country-requests", type=int, default=500)
    parser.add_argument("--min-user-requests", type=int, default=10)
    parser.add_argument("--max-country-requests-per-user", type=int, default=20)
    parser.add_argument("--baseline-repetitions", type=int, default=50)
    parser.add_argument("--seed", type=int, default=19)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = _load_dataset(args.dataset)
    (
        global_counts,
        raw_country_counts,
        user_counts,
        country_user_topics,
        stats,
        examples,
    ) = _collect_counts(dataset, args.batch_size)
    country_counts, country_global_counts = _cap_country_user_contributions(
        country_user_topics, args.max_country_requests_per_user
    )
    country_rows = _group_metrics(country_counts, args.min_country_requests)
    user_rows = _group_metrics(user_counts, args.min_user_requests)
    country_rows.sort(key=lambda row: int(row["requests"]), reverse=True)
    user_rows.sort(key=lambda row: int(row["requests"]), reverse=True)
    if not country_rows or not user_rows:
        raise ValueError("not enough eligible countries or users for locality analysis")

    country_explained = _explained_entropy(
        country_global_counts, country_counts, [row["group"] for row in country_rows]
    )
    user_explained = _explained_entropy(global_counts, user_counts, [row["group"] for row in user_rows])
    country_divergence = np.asarray(
        [
            _jensen_shannon_divergence(country_counts[str(row["group"])], country_global_counts)
            for row in country_rows
        ]
    )
    country_divergence_baseline = _divergence_baseline(
        country_rows, country_global_counts, args.baseline_repetitions, args.seed
    )
    user_baseline = _multinomial_baseline(
        user_rows, global_counts, args.baseline_repetitions, args.seed + 1
    )
    top_countries = [str(row["group"]) for row in country_rows[: args.num_countries]]
    top_users_unsorted = [str(row["group"]) for row in user_rows[: args.num_users]]
    top_users = sorted(
        top_users_unsorted,
        key=lambda group: (TOPICS.index(user_counts[group].most_common(1)[0][0]), -sum(user_counts[group].values())),
    )
    user_aliases = {
        str(row["group"]): f"User {index + 1:04d}"
        for index, row in enumerate(user_rows)
    }

    heatmap_outputs = _render_heatmaps(
        args.output_dir / "wildchat_topic_locality_heatmaps",
        top_countries,
        country_counts,
        top_users,
        user_counts,
        country_global_counts,
        global_counts,
        int(stats["classified_requests"]),
        user_aliases,
    )
    concentration_outputs = _render_concentration(
        args.output_dir / "wildchat_topic_concentration",
        country_divergence,
        user_rows,
        country_divergence_baseline,
        user_baseline,
        country_explained,
        user_explained,
    )
    scatter_outputs = _render_scatter(
        args.output_dir / "wildchat_topic_scatter",
        country_rows,
        country_counts,
        country_global_counts,
        user_rows,
        global_counts,
        args.baseline_repetitions,
        args.seed,
    )
    _write_group_csv(args.output_dir / "country_topic_counts.csv", country_rows, country_counts)
    _write_group_csv(args.output_dir / "user_topic_counts.csv", user_rows, user_counts, user_aliases)
    with (args.output_dir / "topic_examples.json").open("w", encoding="utf-8") as handle:
        json.dump(examples, handle, indent=2, ensure_ascii=True)
        handle.write("\n")

    summary = {
        "dataset": str(args.dataset),
        "method": "English first-user turns classified by weighted intent rules",
        "topics": TOPICS,
        "stats": stats,
        "classification_rate_within_english": stats["classified_requests"] / stats["english_conversations"],
        "global_topic_counts": dict(global_counts),
        "country_global_topic_counts": dict(country_global_counts),
        "raw_geolocated_topic_requests": sum(sum(counts.values()) for counts in raw_country_counts.values()),
        "country_request_weighting": (
            f"Each hashed IP contributes at most {args.max_country_requests_per_user} "
            "request-equivalents per country"
        ),
        "eligible_countries": len(country_rows),
        "eligible_users": len(user_rows),
        "country_explained_topic_entropy": country_explained,
        "user_explained_topic_entropy": user_explained,
        "country_median_dominant_share": float(np.median([row["dominant_share"] for row in country_rows])),
        "user_median_dominant_share": float(np.median([row["dominant_share"] for row in user_rows])),
        "country_median_topic_divergence_bits": float(np.median(country_divergence)),
        "country_null_median_topic_divergence_bits": float(np.median(country_divergence_baseline)),
        "user_null_median_dominant_share": float(np.median(user_baseline)),
        "top_country_names": top_countries,
        "top_user_aliases": [user_aliases[group] for group in top_users],
        "outputs": {
            "heatmaps": heatmap_outputs,
            "concentration": concentration_outputs,
            "scatter": scatter_outputs,
        },
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
