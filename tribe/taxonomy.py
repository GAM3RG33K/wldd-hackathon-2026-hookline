"""Emotion taxonomy for short-form video (shared by the GPU Space and the local fusion).

Categories follow Cowen & Keltner (2017, PNAS), who mapped 27 emotions evoked by short videos;
these are the 20 relevant to reels. Each has circumplex coordinates (arousal 0..1, valence -1..1).

Visual prompts follow appraisal theory: an emotion is evoked by how a viewer appraises the
SITUATION ("an injured animal" -> empathic pain), not by how a frame looks ("a sad picture"),
so every emotion gets situation prompts as well as mood prompts. The Space returns a raw
similarity per prompt; the prompt -> emotion mapping is applied locally (tunable without a GPU).
"""

EMOTIONS = {
    #  name            arousal valence
    "amusement":       (0.70, 0.80),
    "excitement":      (0.90, 0.70),
    "awe":             (0.80, 0.70),
    "admiration":      (0.60, 0.70),
    "adoration":       (0.45, 0.90),   # cute
    "satisfaction":    (0.40, 0.60),   # oddly satisfying
    "craving":         (0.55, 0.50),   # food
    "surprise":        (0.90, 0.20),
    "interest":        (0.55, 0.30),   # curiosity
    "nostalgia":       (0.35, 0.30),
    "romance":         (0.50, 0.80),
    "hope":            (0.55, 0.65),   # hope / relief
    "empathic_pain":   (0.70, -0.60),  # empathy for suffering
    "sadness":         (0.25, -0.75),
    "fear":            (0.90, -0.70),
    "anger":           (0.90, -0.80),  # outrage
    "disgust":         (0.65, -0.80),
    "awkwardness":     (0.60, -0.40),  # cringe
    "moved":           (0.65, 0.60),   # kama muta: kindness, compassion, reunion
    "calm":            (0.10, 0.20),   # baseline / no strong appraisal
}
NAMES = list(EMOTIONS)

# Peaks of these count as strong emotions (high arousal or deep valence).
STRONG = ["amusement", "awe", "anger", "sadness", "moved", "fear", "disgust", "empathic_pain"]
STRONG_LABEL = {"amusement": "laughter", "anger": "outrage", "sadness": "sorrow"}

VISUAL_PROMPTS = {  # prompt -> emotion. Situations (appraisal) + moods, across content types.
    "a person doing something hilarious": "amusement", "a funny prank or blooper": "amusement",
    "a comedic reaction or funny face": "amusement",
    "an exciting fast action moment": "excitement", "a thrilling stunt or sports highlight": "excitement",
    "a breathtaking view of massive scale": "awe", "an incredible, almost impossible feat": "awe",
    "someone showing impressive skill or talent": "admiration", "an expert working flawlessly": "admiration",
    "an adorable baby or puppy": "adoration", "a cute animal being playful": "adoration",
    "an oddly satisfying perfect fit or process": "satisfaction", "something being cleaned or restored": "satisfaction",
    "delicious food being cooked": "craving", "a mouth-watering dish close-up": "craving",
    "an unexpected twist or reveal": "surprise", "a shocking moment": "surprise",
    "someone explaining a secret or a fact": "interest", "a mystery being investigated": "interest",
    "old memories, childhood or retro footage": "nostalgia",
    "a couple in love": "romance", "a romantic moment": "romance",
    "a rescue or a recovery": "hope", "someone overcoming hardship": "hope", "a happy ending": "hope",
    "a person or animal injured or in pain": "empathic_pain", "an animal suffering or struggling": "empathic_pain",
    "someone grieving a loss": "sadness", "a person crying alone": "sadness",
    "a dangerous or scary situation": "fear", "a horror scene": "fear",
    "an injustice, someone being mistreated": "anger", "a heated, angry argument": "anger",
    "something gross or disgusting": "disgust", "rotten, filthy or dirty mess": "disgust",
    "an embarrassing cringe moment": "awkwardness", "an awkward social situation": "awkwardness",
    "an act of kindness or compassion": "moved", "people caring for someone in need": "moved",
    "an emotional reunion": "moved",
    "an ordinary everyday scene": "calm", "a plain product shot": "calm", "a person talking to the camera": "calm",
}

MUSIC_PROMPTS = {  # prompt -> {emotion: weight}; timbre + mood, across genres
    "slow mournful melody on flute or strings": {"sadness": 0.7, "empathic_pain": 0.3},
    "melancholic sad piano music": {"sadness": 1.0},
    "hopeful uplifting inspiring music": {"hope": 0.7, "moved": 0.3},
    "emotional music swelling to a climax": {"moved": 0.6, "awe": 0.4},
    "epic triumphant cinematic orchestra": {"awe": 0.7, "excitement": 0.3},
    "tense suspenseful music": {"fear": 0.6, "interest": 0.4},
    "scary horror soundtrack": {"fear": 1.0},
    "happy upbeat cheerful music": {"amusement": 0.4, "excitement": 0.3, "hope": 0.3},
    "comedic quirky funny music": {"amusement": 1.0},
    "romantic love song": {"romance": 1.0},
    "nostalgic retro music": {"nostalgia": 1.0},
    "energetic hype phonk or edm beat": {"excitement": 1.0},
    "aggressive angry heavy music": {"anger": 0.6, "excitement": 0.4},
    "mysterious curious music": {"interest": 1.0},
    "calm relaxing ambient music": {"calm": 1.0},
    "people talking with no music": {"calm": 1.0},
    "silence": {"calm": 1.0},
}

# GoEmotions (transcript) -> taxonomy
GO_MAP = {
    "amusement": {"amusement": 1.0}, "excitement": {"excitement": 1.0}, "joy": {"excitement": 0.5, "hope": 0.5},
    "admiration": {"admiration": 1.0}, "pride": {"admiration": 0.5, "hope": 0.5}, "love": {"adoration": 0.5, "romance": 0.5},
    "desire": {"craving": 1.0}, "surprise": {"surprise": 1.0}, "realization": {"surprise": 0.5, "interest": 0.5},
    "curiosity": {"interest": 1.0}, "confusion": {"interest": 0.5},
    "optimism": {"hope": 1.0}, "relief": {"hope": 1.0}, "gratitude": {"moved": 1.0}, "caring": {"moved": 1.0},
    "sadness": {"sadness": 1.0}, "grief": {"sadness": 1.0}, "remorse": {"sadness": 0.7},
    "disappointment": {"sadness": 0.6}, "fear": {"fear": 1.0}, "nervousness": {"fear": 0.7},
    "anger": {"anger": 1.0}, "annoyance": {"anger": 0.6}, "disapproval": {"anger": 0.5},
    "disgust": {"disgust": 1.0}, "embarrassment": {"awkwardness": 1.0}, "approval": {"admiration": 0.3},
}

# AudioSet sound events (from the speech-gate AST pass) -> taxonomy
SOUND_MAP = {"amusement": "amusement", "awe": "awe", "surprise": "surprise", "tension": "fear", "sadness": "sadness"}
