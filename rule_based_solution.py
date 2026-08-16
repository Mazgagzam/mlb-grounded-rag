"""Rule-based exact-match solver for the Kaggle MLB JSON QA competition.

The solver never calls an LLM.  It resolves the entity and question context,
then ranks scalar JSON paths using baseball-specific aliases.  Decimal values
are kept as Decimal so trailing zeroes from JSON survive in the submission.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable

import ijson


MONTHS = (
    "january february march april may june july august september october "
    "november december"
).split()

SEASON_WORDS = {
    "preseason": "PRE",
    "spring training": "PRE",
    "spring league": "PRE",
    "postseason": "PST",
    "playoffs": "PST",
    "regular season": "REG",
}

# Longest phrases must be tested first.  A rule maps language to likely leaf
# keys and, optionally, path tokens that disambiguate overloaded leaves.
METRICS: list[tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]] = [
    (("release the slider pitch type",), ("count",), ("pitch_types", "slider", "release_speed")),
    (("rank in the national league",), ("league",), ("rank",)),
    (("games back in the wild card race", "wild card race"), ("wild_card_back",), ()),
    (("division elimination number",), ("division_elimination_number",), ()),
    (("team loss",), ("team_loss",), ()),
    (("strikeout-to-walk ratio",), ("kbb", "k_bb"), ("pitching",)),
    (("ground out to fly out ratio", "ground ball to fly ball ratio", "gofo"), ("gofo",), ()),
    (("average number of pitches per batter faced",), ("per_bf", "p_bf", "pitches_per_bf"), ("pitches",)),
    (("opponent batting average",), ("oba", "avg"), ("pitching",)),
    (("on-base average", "oba"), ("oba",), ("pitching",)),
    (("on-base percentage from singles",), ("s",), ("onbase",)),
    (("on-base percentage", "obp"), ("obp",), ()),
    (("slugging percentage",), ("slg",), ()),
    (("batting average",), ("avg",), ("hitting",)),
    (("winning percentage", "win percentage"), ("win_p", "win_pct"), ("standings",)),
    (("save opportunity", "svo"), ("svo",), ("pitching",)),
    (("games back",), ("games_back", "gb"), ()),
    (("intentional walks",), ("ibb",), ()),
    (("innings pitched", "innings did",), ("ip_1", "ip_2", "ip"), ("pitching",)),
    (("batters faced", "batters did"), ("bf",), ("pitching",)),
    (("batted balls in play",), ("bip",), ()),
    (("swings and misses", "swing strikes"), ("kswing",), ()),
    (("strikeouts from swings",), ("kswing",), ()),
    (("total strikeouts", "strikeout count", "number of strikeouts", "strikeouts did"), ("ktotal", "k"), ()),
    (("earned run average", "era"), ("era",), ("pitching",)),
    (("earned run total", "earned runs"), ("er",), ("pitching",)),
    (("runs allowed",), ("runs", "total"), ("pitching",)),
    (("home run total", "home runs"), ("hr",), ()),
    (("runs batted in", "rbis", "rbi"), ("rbi",), ("hitting",)),
    (("runs did", "runs scored", "hitting runs scored", "total number of runs"), ("runs", "total", "r"), ("hitting",)),
    (("extra-base hits",), ("xbh",), ("hitting",)),
    (("at-bats", "at bats"), ("ab",), ("hitting",)),
    (("foul balls",), ("foul",), ("outcome",)),
    (("balls thrown",), ("ball",), ("outcome",)),
    (("wild pitches",), ("wp",), ("pitching",)),
    (("hit a batter",), ("hbp",), ("pitching",)),
    (("throwing errors",), ("throwing",), ("errors",)),
    (("complete games", "complete their game"), ("complete", "cg"), ("games",)),
    (("games start", "games started"), ("start", "gs"), ()),
    (("games did", "games played", "times did", "play count"), ("play", "games", "ap"), ()),
    (("walks",), ("bb",), ()),
    (("hits",), ("h",), ()),
    (("doubles",), ("d",), ("onbase",)),
    (("blown saves",), ("blown_save", "bs"), ("pitching",)),
    (("earn a save", "earned a save", "saves", "save count"), ("save", "sv"), ("pitching",)),
    (("wins", "games won"), ("win", "wins", "w"), ()),
    (("losses", "games lose", "games lost"), ("loss", "losses", "l"), ()),
    (("putouts",), ("po",), ("fielding",)),
    (("assists",), ("a", "assists"), ("fielding",)),
    (("groundballs", "ground balls"), ("groundball",), ()),
    (("fly balls",), ("flyball",), ()),
    (("babip",), ("babip",), ()),
    (("whip",), ("whip",), ()),
    (("ops",), ("ops",), ("hitting",)),
    (("iso",), ("iso",), ("hitting",)),
    (("fielding range factor",), ("rf",), ("fielding",)),
    (("fielding statistic for innings pitched",), ("inn_1",), ("fielding",)),
    (("overall fielding statistic",), ("inn_1",), ("fielding",)),
    (("pitching value",), ("value",), ("pitching",)),
    (("pitch type",), ("count", "total"), ("pitch",)),
    (("lineup order",), ("order",), ("lineup",)),
    (("caught stealing",), ("cs",), ("steal",)),
    (("number of pitches thrown", "total number of pitches thrown"), ("pitch_count",), ()),
    (("k/9",), ("k9", "k_9"), ("pitching",)),
    (("jersey number",), ("jersey_number",), ()),
    (("reference number",), ("reference",), ()),
    (("primary position",), ("primary_position",), ()),
    (("position",), ("position",), ()),
    (("pitcher hand",), ("throw_hand",), ()),
    (("batting hand",), ("bat_hand",), ()),
    (("birthdate",), ("birthdate",), ()),
    (("height",), ("height",), ()),
    (("salary",), ("salary",), ()),
    (("abbreviation",), ("abbr", "alias"), ()),
    (("fight song",), ("fight_song",), ()),
    (("minor league affiliates",), ("minorleague_affiliate",), ()),
    (("retired numbers",), ("retired_numbers",), ()),
    (("mascot",), ("mascot",), ()),
    (("owner",), ("owner",), ()),
    (("president",), ("president",), ()),
    (("founded",), ("founded",), ()),
    (("playoff appearances",), ("playoff_appearances",), ()),
    (("division titles",), ("division_titles",), ()),
    (("championship",), ("championship_seasons",), ()),
    (("surface",), ("surface",), ()),
    (("venue for", "venue city", "what city"), ("city",), ("venue",)),
    (("market",), ("market",), ()),
    (("update date",), ("updated", "update_date"), ("injur",)),
    (("start date",), ("start_date",), ("injur",)),
    (("assignment",), ("assignment",), ("official",)),
    (("rank",), ("rank",), ()),
]


def norm(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    value = value.lower().replace("’", "'")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def key_norm(value: str) -> str:
    return norm(value).replace(" ", "_")


def scalar_text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value)


def flatten(value: Any, path: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield from flatten(child, path + (str(key),))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from flatten(child, path + (str(index),))
    else:
        yield path, value


@dataclass
class Entity:
    kind: str
    key: str
    name: str
    data: dict[str, Any]


@dataclass
class Candidate:
    score: float
    path: tuple[str, ...]
    value: Any
    source: str


class Solver:
    def __init__(self, data_dir: Path, questions: list[str]):
        self.data_dir = data_dir
        self.questions_norm = [norm(q) for q in questions]
        self.question_blob = "\n".join(self.questions_norm)
        self.teams = self._load_json(data_dir / "teams.json")
        self.league = self._load_json(data_dir / "league.json")
        self.team_entities = self._make_team_entities()
        self.players = self._load_relevant_players(data_dir / "players.json")

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle, parse_float=Decimal)

    def _make_team_entities(self) -> list[Entity]:
        result = []
        for key, data in self.teams.items():
            profile = data.get("data", {}).get("team_profile", {})
            market, name = profile.get("market", ""), profile.get("name", "")
            result.append(Entity("team", key, f"{market} {name}".strip(), data))
        return result

    def _load_relevant_players(self, path: Path) -> list[Entity]:
        result: list[Entity] = []
        with path.open("rb") as handle:
            for key, data in ijson.kvitems(handle, "", use_float=False):
                profile = data.get("data", {}).get("player_profile", {}).get("player", {})
                full_name = profile.get("full_name") or key.replace("_", " ")
                variants = {norm(full_name), norm(key.replace("_", " "))}
                variants |= {re.sub(r" (jr|sr|ii|iii|iv)$", "", x) for x in variants}
                first = norm(str(profile.get("first_name", "")))
                last = norm(str(profile.get("last_name", "")))
                fuzzy_name = bool(
                    len(first) >= 4 and len(last) >= 3
                    and re.search(rf"\b{re.escape(first[:4])}[a-z]*\s+{re.escape(last)}\b", self.question_blob)
                )
                if fuzzy_name or any(len(x) >= 5 and x in self.question_blob for x in variants):
                    result.append(Entity("player", key, full_name, data))
        return result

    def _entities_in_question(self, question: str) -> tuple[list[Entity], list[Entity]]:
        qn = norm(question)
        players = []
        for entity in self.players:
            profile = entity.data.get("data", {}).get("player_profile", {}).get("player", {})
            first = norm(str(profile.get("first_name", "")))
            last = norm(str(profile.get("last_name", "")))
            fuzzy = bool(
                len(first) >= 4 and len(last) >= 3
                and re.search(rf"\b{re.escape(first[:4])}[a-z]*\s+{re.escape(last)}\b", qn)
            )
            if fuzzy or norm(entity.name) in qn or norm(entity.key.replace("_", " ")) in qn:
                players.append(entity)
        teams = []
        for entity in self.team_entities:
            profile = entity.data.get("data", {}).get("team_profile", {})
            names = {
                norm(entity.name),
                norm(profile.get("name", "")),
                norm(entity.key.replace("_", " ")),
            }
            # A nickname alone must be reasonably distinctive.
            if any(len(name) >= 4 and re.search(rf"\b{re.escape(name)}\b", qn) for name in names if name):
                teams.append(entity)
        listed_tail = qn.split("as listed in", 1)[1] if "as listed in" in qn else ""
        teams.sort(key=lambda e: (
            0 if listed_tail and (norm(e.name) in listed_tail or norm(e.data.get("data", {}).get("team_profile", {}).get("name", "")) in listed_tail) else 1,
            qn.find(norm(e.name)) if norm(e.name) in qn else 10_000,
        ))
        return players, teams

    @staticmethod
    def _metric_rule(qn: str) -> tuple[tuple[str, ...], tuple[str, ...], str]:
        if re.search(r"\bgames did .+\bstart\b", qn):
            return ("start", "gs"), (), "games did start"
        if re.search(r"\bhow many batters did .+\bwalk\b", qn):
            return ("bb",), ("pitching",), "batters walked"
        for phrases, leaves, path_hints in METRICS:
            for phrase in phrases:
                normalized_phrase = norm(phrase)
                if re.search(rf"\b{re.escape(normalized_phrase)}\b", qn):
                    return leaves, path_hints, phrase
        if qn.startswith("did ") and " play " in f" {qn} ":
            return ("play", "games"), (), "did play"
        if re.search(r"\bwon\b", qn):
            return ("win", "wins", "w"), (), "won"
        if re.search(r"\b(?:lose|lost|loss|losses|incur|incurred)\b", qn):
            return ("loss", "losses", "l"), (), "loss"
        if re.search(r"\bstart(?:ed)?\b", qn) and "game" in qn:
            return ("start", "gs"), (), "started"
        return (), (), ""

    @staticmethod
    def _relative_label(qn: str) -> str | None:
        if "previous match" in qn or "last appearance" in qn or "last game" in qn:
            return "previous_match"
        words = {"two": 2, "three": 3, "four": 4, "five": 5}
        match = re.search(r"(?:game |played |game played |game that took place )?(\d+|two|three|four|five) matches ago", qn)
        if match:
            number = int(match.group(1)) if match.group(1).isdigit() else words[match.group(1)]
            return f"{number}_matches_ago"
        return None

    def _context(self, question: str, teams: list[Entity]) -> dict[str, Any]:
        qn = norm(question)
        years = re.findall(r"\b(?:19|20)\d{2}\b", qn)
        month = next((m for m in MONTHS if re.search(rf"\b{m}\b", qn)), None)
        season = next((code for phrase, code in SEASON_WORDS.items() if phrase in qn), None)
        if season is None and years and "season" in qn:
            season = "REG"
        role = None
        if "starter" in qn or "starting pitcher" in qn:
            role = "starters"
        elif "bullpen" in qn or "reliever" in qn:
            role = "bullpen"
        elif "pitching overall" in qn:
            role = "overall"
        split_qn = re.sub(r"\bhome runs?\b", "", qn)
        split = next((x for x in ("home", "away", "day", "night", "grass", "turf") if re.search(rf"\b{x}\b", split_qn)), None)
        handed = None
        if "right handed hitter" in qn or "right handed hitters" in qn:
            handed = ("hitter_hand", "r")
        elif "left handed hitter" in qn or "left handed hitters" in qn:
            handed = ("hitter_hand", "l")
        elif "right handed pitcher" in qn or "right handed pitchers" in qn:
            handed = ("pitcher_hand", "r")
        elif "left handed pitcher" in qn or "left handed pitchers" in qn:
            handed = ("pitcher_hand", "l")
        field_position = None
        if "first base" in qn:
            field_position = "1b"
        elif "shortstop" in qn:
            field_position = "ss"
        elif "as a catcher" in qn:
            field_position = "c"
        relative = self._relative_label(qn)
        opponent = None
        for team in teams:
            tname = norm(team.name)
            nickname = norm(team.data.get("data", {}).get("team_profile", {}).get("name", ""))
            if re.search(rf"\bagainst (?:the )?(?:{re.escape(tname)}|{re.escape(nickname)})\b", qn):
                opponent = key_norm(team.key)
                break
        if opponent is None and "as listed in" in qn and len(teams) > 1:
            opponent = key_norm(teams[1].key)
        venues = []
        for team in self.team_entities:
            venue = team.data.get("data", {}).get("team_profile", {}).get("venue", {})
            for label in (venue.get("name", ""), f"{venue.get('market', '')} {venue.get('name', '')}"):
                if label and norm(label) in qn:
                    venues.append((key_norm(label), key_norm(venue.get("name", "")), key_norm(team.key)))
        return {"years": years, "month": month, "season": season, "role": role,
                "split": split, "handed": handed, "field_position": field_position,
                "relative": relative, "opponent": opponent, "venues": venues, "implicit_year": False}

    def _path_score(
        self,
        path: tuple[str, ...],
        value: Any,
        leaves: tuple[str, ...],
        hints: tuple[str, ...],
        context: dict[str, Any],
        entity: Entity,
        qn: str,
    ) -> float:
        seg = tuple(key_norm(x) for x in path)
        leaf = seg[-1] if seg else ""
        joined = "/".join(seg)
        if "_comment" in seg or isinstance(value, (dict, list)):
            return -1e9
        score = 0.0
        if leaves:
            if leaf in leaves:
                score += 100 - leaves.index(leaf) * 3
            else:
                return -1e9
        for hint in hints:
            score += 12 if hint in seg or hint in joined else -5
        if entity.kind == "league" and context.get("season") == "PRE" and "standings" in joined:
            score += 70

        # Strong context constraints.  Missing a stated context is usually a
        # wrong aggregate even when the leaf metric matches.
        for year in context["years"]:
            score += 30 if year in seg else -45
        if context["season"]:
            score += 24 if context["season"].lower() in seg else -32
        if context["month"]:
            score += 28 if context["month"] in seg else -38
        if context["relative"]:
            score += 45 if context["relative"] in seg else -65
        if context["role"]:
            score += 20 if context["role"] in seg else -18
        if context["split"]:
            score += 16 if context["split"] in seg else -10
        if context.get("handed"):
            hand_group, hand = context["handed"]
            score += 28 if hand_group in seg and hand in seg else -30
        if context.get("field_position"):
            position = context["field_position"]
            score += 30 if "positions" in seg and position in seg else -18
        if context["opponent"]:
            score += 30 if context["opponent"] in joined else -22
        if context["venues"]:
            found = any(any(v and v in joined for v in variants) for variants in context["venues"])
            score += 30 if found else -22

        primary_side = context.get("primary_side")
        if primary_side and ("home" in seg or "away" in seg):
            score += 26 if primary_side in seg else -26

        # For a player's game summary, prefer that player's own node instead
        # of the identically named team aggregate statistic.
        if entity.kind == "player" and context["relative"]:
            profile = entity.data.get("data", {}).get("player_profile", {}).get("player", {})
            first = key_norm(str(profile.get("first_name", "")))
            last = key_norm(str(profile.get("last_name", "")))
            exactish = key_norm(entity.key) in joined or any(
                first[:4] and last and first[:4] in part and last in part for part in seg
            )
            score += 42 if exactish else 0
            score += 30  # prefer the player's own game snapshot to a team fallback
        # Named objects (players, officials, teams) are represented as path
        # segments. Exact phrase overlap is a powerful disambiguator.
        named_overlap = 0
        for part in seg:
            phrase = norm(part)
            if len(phrase) >= 7 and " " in phrase and re.search(rf"\b{re.escape(phrase)}\b", qn):
                named_overlap += 22
        score += min(named_overlap, 66)
        if "lineup" in qn:
            score += 35 if "lineup" in seg else -28
            target_index = None
            ordinal = re.search(r"\b(\d+)(?:st|nd|rd|th) player\b", qn)
            if ordinal:
                target_index = int(ordinal.group(1)) - 1
            elif "batting second" in qn:
                target_index = 1
            else:
                position_number = re.search(r"\bposition (\d+)\b", qn)
                if position_number:
                    target_index = int(position_number.group(1)) - 1
            if target_index is not None:
                score += 32 if str(target_index) in seg else -8
        if "first inning" in qn:
            score += 18 if "first" in seg or "1" in seg else -8
        if "overall" in qn:
            score += 7 if "overall" in seg else 0
        if "hitting" in qn:
            score += 8 if "hitting" in seg else -3
        if "pitch" in qn or "pitcher" in qn or "allow" in qn:
            score += 5 if "pitching" in seg else 0
        if entity.kind == "player":
            profile = entity.data.get("data", {}).get("player_profile", {}).get("player", {})
            if profile.get("position") == "P" and any(leaf_key in leaves for leaf_key in ("ktotal", "k", "bb")):
                score += 28 if "pitching" in seg else -12
        if context.get("implicit_year"):
            score += 3 if "2025" in seg else 2 if "2024" in seg else 1 if "2023" in seg else 0
        # Prefer shorter paths only as a final tie-breaker.
        score -= len(path) * 0.02
        return score

    @staticmethod
    def _subtrees(entity: Entity, context: dict[str, Any], qn: str) -> list[tuple[tuple[str, ...], Any]]:
        """Choose the smallest branch that can contain the answer.

        Game summaries are very large, so avoiding unrelated seasons/games is
        the main runtime optimization.
        """
        data = entity.data.get("data", entity.data)
        relative = context["relative"]
        if entity.kind == "player":
            if relative:
                branch = data.get("last_10_games", {}).get(relative.replace("_", " "))
                return [(("data", "last_10_games", relative.replace("_", " ")), branch)] if branch else []
            player = data.get("player_profile", {}).get("player", {})
            if context["years"]:
                result = []
                seasons = player.get("seasons", {})
                for year in context["years"]:
                    branch = seasons.get(year)
                    if branch is not None:
                        result.append((("data", "player_profile", "player", "seasons", year), branch))
                return result
            if context.get("implicit_year"):
                seasons = player.get("seasons", {})
                return [
                    (("data", "player_profile", "player", "seasons", year), seasons[year])
                    for year in ("2025", "2024", "2023") if year in seasons
                ]
            # Static biographical fields only; seasons would add millions of
            # irrelevant candidates.
            static = {k: v for k, v in player.items() if k != "seasons"}
            return [(('data', 'player_profile', 'player'), static)]

        if entity.kind == "team":
            if relative:
                branch = data.get("last_10_games", {}).get(relative.replace("_", " "))
                return [(("data", "last_10_games", relative.replace("_", " ")), branch)] if branch else []
            if context["years"]:
                result = []
                for year in context["years"]:
                    branch = data.get(year)
                    if branch is not None:
                        result.append((("data", year), branch))
                return result
            if context.get("implicit_year"):
                return [(("data", year), data[year]) for year in ("2025", "2024", "2023") if year in data]
            return [(('data', 'team_profile'), data.get('team_profile', {}))]

        return [((), entity.data)]

    def answer(self, question: str, top_n: int = 5) -> tuple[str, list[Candidate]]:
        qn = norm(question)
        players, teams = self._entities_in_question(question)
        leaves, hints, _ = self._metric_rule(qn)
        context = self._context(question, teams)

        static_leaves = {
            "position", "primary_position", "jersey_number", "reference", "birthdate", "height",
            "salary", "abbr", "alias", "fight_song", "minorleague_affiliate", "retired_numbers",
            "mascot", "owner", "president", "founded", "playoff_appearances", "division_titles",
            "championship_seasons", "surface", "city", "market", "updated", "update_date", "start_date",
            "throw_hand", "bat_hand",
        }
        # The snapshot is from the 2025 season. Questions that ask for a split
        # but omit the year refer to that current season.
        if leaves and not context["years"] and not context["relative"] and not set(leaves) <= static_leaves:
            context["implicit_year"] = True
            context["season"] = context["season"] or "REG"
        if leaves and context["years"] and not context["season"] and context["month"] != "march" and not set(leaves) <= static_leaves:
            context["season"] = "REG"
        if set(leaves) <= {"throw_hand", "bat_hand"}:
            context["handed"] = None

        # Standings store home/away records as compound scalar keys.
        if context["split"] in ("home", "away"):
            side = context["split"]
            if any(x in leaves for x in ("win", "wins", "w")):
                leaves = (f"{side}_win",) + leaves
            elif any(x in leaves for x in ("loss", "losses", "l")):
                leaves = (f"{side}_loss",) + leaves

        sources: list[Entity] = []
        if players:
            sources.extend(players)
            if context["relative"]:
                # A player's own game branch can be incomplete. The opponent's
                # team summary often still contains that player's line.
                sources.extend(teams)
        elif teams:
            sources.append(teams[0])
        else:
            sources.append(Entity("league", "league", "league", self.league))
        # Standings and injuries are centralized in league.json; include it as
        # a fallback even when a team/player is named.
        if any(x in qn for x in ("standings", "games back", "elimination", "rank", "injur", "winning percentage", "win percentage")) or context["season"] == "PRE":
            sources.append(Entity("league", "league", "league", self.league))

        candidates: list[Candidate] = []
        for entity in sources:
            for prefix, subtree in self._subtrees(entity, context, qn):
                local_context = dict(context)
                if entity.kind == "team" and context["relative"] and isinstance(subtree, dict):
                    profile = entity.data.get("data", {}).get("team_profile", {})
                    team_id = str(profile.get("id", ""))
                    team_abbr = str(profile.get("abbr", ""))
                    for side in ("home", "away"):
                        side_data = subtree.get(side, {})
                        if team_id and side_data.get("id") == team_id or team_abbr and side_data.get("abbr") == team_abbr:
                            local_context["primary_side"] = side
                            break
                    # Lineup is a dictionary keyed by fielding position, while
                    # batting order is stored in each value. Resolve sibling
                    # fields explicitly rather than confusing dict keys with
                    # list indexes.
                    side = local_context.get("primary_side")
                    game = subtree.get("summary", {}).get("game", {})
                    lineup = game.get(side, {}).get("lineup", {}) if side else {}
                    if lineup and "lineup" in qn:
                        direct_key = None
                        direct_leaf = None
                        if "lineup order" in qn:
                            match = re.search(r"\bposition (\d+)\b", qn)
                            if match:
                                wanted = int(match.group(1))
                                direct_key = next((k for k, item in lineup.items() if item.get("position") == wanted), None)
                                direct_leaf = "order"
                        else:
                            wanted_order = None
                            match = re.search(r"\b(\d+)(?:st|nd|rd|th) player\b", qn)
                            if match:
                                wanted_order = int(match.group(1)) - 1
                            elif "batting second" in qn:
                                wanted_order = 1
                            if wanted_order is not None:
                                direct_key = next((k for k, item in lineup.items() if item.get("order") == wanted_order), None)
                                direct_leaf = "position"
                        if direct_key is not None and direct_leaf is not None:
                            direct_path = prefix + ("summary", "game", side, "lineup", direct_key, direct_leaf)
                            candidates.append(Candidate(1000.0, direct_path, lineup[direct_key][direct_leaf], entity.name))
                for path, value in flatten(subtree, prefix):
                    score = self._path_score(path, value, leaves, hints, local_context, entity, qn)
                    if score > -1e8:
                        candidates.append(Candidate(score, path, value, entity.name))
        candidates.sort(key=lambda c: c.score, reverse=True)
        if not candidates:
            return "0", []
        best = candidates[0]
        return scalar_text(best.value), candidates[:top_n]


def run(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    with Path(args.test).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    solver = Solver(data_dir, [row["question"] for row in rows])

    output_rows = []
    diagnostic_rows = []
    for row in rows:
        answer, candidates = solver.answer(row["question"], args.top_n)
        output_rows.append({"ID": row["ID"], "ANSWER": answer})
        best = candidates[0] if candidates else None
        second = candidates[1] if len(candidates) > 1 else None
        diagnostic_rows.append({
            "ID": row["ID"],
            "question": row["question"],
            "ANSWER": answer,
            "score": f"{best.score:.2f}" if best else "",
            "margin": f"{best.score - second.score:.2f}" if best and second else "",
            "source": best.source if best else "",
            "path": "/".join(best.path) if best else "",
            "alternatives": " || ".join(
                f"{scalar_text(c.value)} @ {'/'.join(c.path)} [{c.score:.1f}]" for c in candidates[1:]
            ),
        })

    with Path(args.output).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("ID", "ANSWER"))
        writer.writeheader()
        writer.writerows(output_rows)
    with Path(args.diagnostics).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=diagnostic_rows[0].keys())
        writer.writeheader()
        writer.writerows(diagnostic_rows)
    print(f"Wrote {len(output_rows)} answers to {args.output}")
    print(f"Diagnostics: {args.diagnostics}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--test", default="test.csv")
    parser.add_argument("--output", default="submission_rule_based.csv")
    parser.add_argument("--diagnostics", default="diagnostics.csv")
    parser.add_argument("--top-n", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
