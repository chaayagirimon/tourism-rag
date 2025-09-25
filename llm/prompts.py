"""Template-based benchmark query generator.

Produces queries by combining a small set of templates with actual POI names
from the cards corpus. This keeps everything local/offline.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

DEFAULT_CARDS_PATH = Path("./data/poi_cards.cleaned.json")


class PromptGenerationError(RuntimeError):
    pass


@dataclass
class Template:
    intent: str
    pattern: str

    def format(self, **kwargs) -> str:
        return self.pattern.format(**kwargs)


TEMPLATES: List[Template] = [
    Template("food_traditional", "traditional {cuisine} spot in {city_name} near {landmark}"),
    Template("food_sustainable", "sustainable cafe in {city_name} featuring {feature}"),
    Template("museum_kids", "kid-friendly museum in {city_name} with {feature}"),
    Template("museum_history", "history museum in {city_name} focusing on {feature}"),
    Template("outdoor", "scenic outdoor place in {city_name} for {activity}"),
    Template("nightlife", "nightlife venue in {city_name} known for {feature}"),
    Template("lodging", "boutique stay in {city_name} near {landmark}"),
    Template("accessibility", "accessible attraction in {city_name} with {feature}"),
    Template("hidden_gem", "hidden gem in {city_name} loved by locals"),
]


def load_city_index(cards_path: Path | str = DEFAULT_CARDS_PATH) -> Dict[str, List[Dict]]:
    path = Path(cards_path)
    if not path.exists():
        raise FileNotFoundError(f"Cards file not found: {path}")

    city_to_cards: Dict[str, List[Dict]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            card = obj.get("card", obj)
            city = (card.get("city") or "").strip().lower()
            if not city:
                continue
            city_to_cards.setdefault(city, []).append(card)
    return city_to_cards


def list_cities(cards_path: Path | str = DEFAULT_CARDS_PATH) -> List[str]:
    return sorted(load_city_index(cards_path).keys())


def _choose(values: Sequence[str], rng: random.Random) -> str:
    if not values:
        return ""
    return rng.choice(values)


def _derive_features(card: Dict) -> Dict[str, str]:
    cuisine = card.get("cuisine") or card.get("kind") or "local cuisine"
    landmark = card.get("landmark") or card.get("name") or "the city center"
    feature = card.get("feature") or _choose(card.get("wv") or [], random.Random()) or "local culture"
    activity = card.get("activity") or "relaxation"
    return {
        "cuisine": cuisine,
        "landmark": landmark,
        "feature": feature,
        "activity": activity,
    }


def generate_benchmark_queries(
    *,
    cities: Sequence[str],
    per_city: int = 10,
    cards_path: Path | str = DEFAULT_CARDS_PATH,
    seed: int = 0,
) -> List[Dict[str, object]]:
    city_cards = load_city_index(cards_path)
    rng = random.Random(seed)
    prompts: List[Dict[str, object]] = []

    for city in cities:
        cards = city_cards.get(city.lower())
        if not cards:
            raise PromptGenerationError(f"No cards found for city: {city}")
        for _ in range(per_city):
            card = rng.choice(cards)
            template = rng.choice(TEMPLATES)
            features = _derive_features(card)
            q = template.format(city_name=city.title(), **features)
            name_contains = card.get("name") or features["landmark"]
            prompts.append({
                "q": q,
                "city": city.lower(),
                "must": {"name_contains": name_contains},
            })
    return prompts
