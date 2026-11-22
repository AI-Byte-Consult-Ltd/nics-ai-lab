"""Classes of NICS Vision Geo, how raw specimen names map to them, and what each may indicate.

Every class has:
  group     -- coarse group, trained as a second head (helps when a class has few images)
  match     -- how a raw specimen name maps here: exact names and/or name-contains words
  commons   -- Wikimedia Commons categories to collect from (on the laptop)
  indicates -- what finding it MAY point to (rules from general geology, not a model output)
  tests     -- simple field checks a person can do to confirm or reject the guess

A raw name that matches no class is not used for training (it stays in the manifest as
`class: null`, so the mapping can be widened later without re-downloading anything).
Classes with too few images after mapping are dropped at build time (config min_images),
so this list can be wider than what a given model is trained on.

The texts are deliberately careful: a photo is a hint, an assay is proof.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

GROUPS = (
    "native_element", "sulfide", "oxide", "carbonate", "silicate", "sulfate_other",
    "igneous_rock", "sedimentary_rock", "metamorphic_rock", "hydrothermal_rock", "meteorite",
)


@dataclass(frozen=True)
class GeoClass:
    key: str
    group: str
    ru: str
    en: str
    exact: tuple[str, ...] = ()
    contains: tuple[str, ...] = ()
    commons: tuple[str, ...] = ()
    indicates: tuple[str, ...] = ()   # keys into INDICATORS
    tests: tuple[str, ...] = ()       # keys into TESTS
    caution: tuple[str, ...] = ()     # keys into CAUTIONS


C = GeoClass
CLASSES: tuple[GeoClass, ...] = (
    # native elements
    C("gold", "native_element", "Самородное золото", "Native gold", exact=("gold",),
      commons=("Gold_(mineral)", "Gold_nuggets"), indicates=("gold",), tests=("malleable", "streak_gold", "heavy")),
    C("silver", "native_element", "Самородное серебро", "Native silver", exact=("silver",),
      commons=("Silver_(mineral)",), indicates=("silver",), tests=("malleable", "heavy")),
    C("copper", "native_element", "Самородная медь", "Native copper", exact=("copper",),
      commons=("Native_copper",), indicates=("copper_primary",), tests=("malleable", "heavy")),
    C("sulfur", "native_element", "Самородная сера", "Native sulfur", exact=("sulfur", "sulphur"),
      commons=("Sulfur_(mineral)",), indicates=("volcanic",), tests=("soft", "smell_sulfur")),
    C("diamond", "native_element", "Алмаз", "Diamond", exact=("diamond",),
      commons=("Diamonds_(mineral)",), indicates=("diamond",), tests=("very_hard",)),
    # sulfides
    C("pyrite", "sulfide", "Пирит", "Pyrite", exact=("pyrite", "marcasite"),
      commons=("Pyrite",), indicates=("sulfide", "gold_associate"), tests=("streak_black", "brittle", "not_gold")),
    C("chalcopyrite", "sulfide", "Халькопирит", "Chalcopyrite", exact=("chalcopyrite", "bornite", "chalcocite"),
      commons=("Chalcopyrite", "Bornite"), indicates=("copper_primary", "sulfide"), tests=("streak_black", "brittle", "not_gold")),
    C("galena", "sulfide", "Галенит", "Galena", exact=("galena",),
      commons=("Galena",), indicates=("lead_zinc",), tests=("cubic_cleavage", "heavy", "streak_grey")),
    C("sphalerite", "sulfide", "Сфалерит", "Sphalerite", exact=("sphalerite",),
      commons=("Sphalerite",), indicates=("lead_zinc",), tests=("streak_brown",)),
    C("pyrrhotite", "sulfide", "Пирротин", "Pyrrhotite", exact=("pyrrhotite",),
      commons=("Pyrrhotite",), indicates=("sulfide", "nickel"), tests=("magnetic_weak", "streak_black")),
    C("silver_sulfide", "sulfide", "Серебряные сульфиды", "Silver sulfides",
      exact=("acanthite", "argentite", "dyscrasite", "proustite", "pyrargyrite"),
      commons=("Acanthite", "Proustite"), indicates=("silver",), tests=("heavy",)),
    # oxides
    C("hematite", "oxide", "Гематит", "Hematite", exact=("hematite", "goethite", "limonite"),
      commons=("Hematite", "Goethite"), indicates=("iron", "gossan"), tests=("streak_red",)),
    C("magnetite", "oxide", "Магнетит", "Magnetite", exact=("magnetite", "chromite"),
      commons=("Magnetite",), indicates=("iron",), tests=("magnetic",)),
    C("corundum", "oxide", "Корунд (рубин, сапфир)", "Corundum (ruby, sapphire)", exact=("corundum", "ruby", "sapphire"),
      commons=("Corundum",), indicates=("gem",), tests=("very_hard",)),
    C("cassiterite", "oxide", "Касситерит", "Cassiterite", exact=("cassiterite",),
      commons=("Cassiterite",), indicates=("tin",), tests=("heavy", "streak_white")),
    C("quartz", "oxide", "Кварц", "Quartz",
      exact=("quartz", "amethyst", "smoky quartz", "rock crystal", "citrine", "chalcedony", "agate", "jasper"),
      commons=("Quartz", "Amethyst", "Agate"), indicates=("quartz_vein",), tests=("hard_scratches_glass", "no_cleavage")),
    C("opal", "oxide", "Опал", "Opal", exact=("opal",), commons=("Opal",), indicates=("gem",), tests=()),
    # carbonates
    C("calcite", "carbonate", "Кальцит", "Calcite", exact=("calcite", "aragonite", "dolomite"),
      commons=("Calcite",), indicates=(), tests=("acid_fizz", "soft_knife")),
    C("copper_carbonate", "carbonate", "Малахит, азурит", "Malachite, azurite", exact=("malachite", "azurite"),
      commons=("Malachite", "Azurite"), indicates=("copper_oxide_zone",), tests=("acid_fizz",)),
    C("lead_zinc_secondary", "carbonate", "Вторичные минералы свинца и цинка", "Secondary lead and zinc minerals",
      exact=("smithsonite", "cerussite", "wulfenite", "pyromorphite", "hemimorphite", "anglesite", "mimetite", "vanadinite"),
      commons=("Wulfenite", "Pyromorphite", "Smithsonite"), indicates=("lead_zinc",), tests=("heavy",)),
    C("rhodochrosite", "carbonate", "Родохрозит", "Rhodochrosite", exact=("rhodochrosite",),
      commons=("Rhodochrosite",), indicates=("manganese",), tests=("acid_fizz",)),
    # silicates
    C("chrysocolla", "silicate", "Хризоколла", "Chrysocolla", exact=("chrysocolla", "dioptase"),
      commons=("Chrysocolla",), indicates=("copper_oxide_zone",), tests=("soft_knife",)),
    C("feldspar", "silicate", "Полевой шпат", "Feldspar",
      exact=("orthoclase", "microcline", "albite", "labradorite", "amazonite", "oligoclase", "sanidine", "anorthite"),
      commons=("Feldspar", "Microcline", "Labradorite"), indicates=(), tests=("two_cleavages",)),
    C("mica", "silicate", "Слюда", "Mica", exact=("muscovite", "biotite", "phlogopite", "lepidolite"),
      commons=("Muscovite", "Biotite"), indicates=("not_gold_flakes",), tests=("splits_in_sheets",)),
    C("garnet", "silicate", "Гранат", "Garnet",
      exact=("garnet", "andradite", "grossular", "almandine", "pyrope", "spessartine", "uvarovite"),
      commons=("Garnets",), indicates=("metamorphic",), tests=("hard_scratches_glass",)),
    C("beryl", "silicate", "Берилл (изумруд, аквамарин)", "Beryl (emerald, aquamarine)",
      exact=("beryl", "emerald", "aquamarine", "morganite", "heliodor"),
      commons=("Beryl",), indicates=("pegmatite", "gem"), tests=("very_hard",)),
    C("tourmaline", "silicate", "Турмалин", "Tourmaline", exact=("tourmaline", "elbaite", "schorl", "dravite", "liddicoatite"),
      commons=("Tourmaline",), indicates=("pegmatite",), tests=("hard_scratches_glass",)),
    C("topaz", "silicate", "Топаз", "Topaz", exact=("topaz",), commons=("Topaz",), indicates=("pegmatite", "gem"),
      tests=("very_hard",)),
    C("zeolite", "silicate", "Цеолиты", "Zeolites",
      exact=("stilbite", "heulandite", "chabazite", "analcime", "natrolite", "mesolite", "scolecite", "thomsonite",
             "laumontite", "apophyllite", "prehnite"),
      commons=("Zeolites",), indicates=("volcanic",), tests=("soft_knife",)),
    C("olivine_pyroxene", "silicate", "Оливин, пироксен", "Olivine, pyroxene",
      exact=("forsterite", "olivine", "peridot", "diopside", "augite", "enstatite"),
      commons=("Olivine",), indicates=("mafic",), tests=()),
    # sulfates and others
    C("fluorite", "sulfate_other", "Флюорит", "Fluorite", exact=("fluorite",), commons=("Fluorite",),
      indicates=("hydrothermal",), tests=("cubic_cleavage", "soft_knife")),
    C("baryte", "sulfate_other", "Барит", "Baryte", exact=("baryte", "barite", "celestine"), commons=("Baryte",),
      indicates=("hydrothermal",), tests=("heavy",)),
    C("gypsum", "sulfate_other", "Гипс", "Gypsum", exact=("gypsum", "selenite"), commons=("Gypsum",),
      indicates=("evaporite",), tests=("fingernail",)),
    C("apatite", "sulfate_other", "Апатит", "Apatite", exact=("fluorapatite", "apatite", "hydroxylapatite"),
      commons=("Apatite",), indicates=("phosphate",), tests=()),
    # rocks
    C("granite", "igneous_rock", "Гранит", "Granite", contains=("granite", "granodiorite", "pegmatite", "syenite"),
      commons=("Granite",), indicates=(), tests=("crystals_visible",)),
    C("diorite_gabbro", "igneous_rock", "Диорит, габбро", "Diorite, gabbro", contains=("diorite", "gabbro", "norite"),
      commons=("Gabbro", "Diorite"), indicates=("mafic",), tests=("crystals_visible",)),
    C("basalt", "igneous_rock", "Базальт", "Basalt", contains=("basalt", "dolerite", "diabase", "scoria"),
      commons=("Basalt",), indicates=("volcanic",), tests=("fine_grained_dark",)),
    C("rhyolite_andesite", "igneous_rock", "Риолит, андезит, дацит", "Rhyolite, andesite, dacite",
      contains=("rhyolite", "andesite", "dacite", "trachyte", "tuff", "ignimbrite"),
      commons=("Rhyolite", "Andesite"), indicates=("volcanic", "epithermal"), tests=()),
    C("volcanic_glass", "igneous_rock", "Вулканическое стекло, пемза", "Volcanic glass, pumice",
      contains=("obsidian", "pumice", "pitchstone", "perlite"), commons=("Obsidian", "Pumice"),
      indicates=("volcanic",), tests=("conchoidal",)),
    C("ultramafic", "igneous_rock", "Перидотит, серпентинит", "Peridotite, serpentinite",
      contains=("peridotite", "serpentin", "dunite", "kimberlite", "pyroxenite", "lherzolite", "harzburgite"),
      commons=("Serpentinite", "Peridotite"), indicates=("nickel_chrome_pgm",), tests=(), caution=("asbestos",)),
    C("sandstone", "sedimentary_rock", "Песчаник", "Sandstone", contains=("sandstone", "quartzite", "arkose", "greywacke"),
      commons=("Sandstone",), indicates=(), tests=("gritty",)),
    C("limestone", "sedimentary_rock", "Известняк", "Limestone", contains=("limestone", "chalk", "travertine", "tufa", "dolostone"),
      commons=("Limestone",), indicates=(), tests=("acid_fizz",)),
    C("shale_mudstone", "sedimentary_rock", "Сланец глинистый, аргиллит", "Shale, mudstone",
      contains=("shale", "mudstone", "siltstone", "claystone"), commons=("Shale",), indicates=("oil_source",), tests=()),
    C("conglomerate", "sedimentary_rock", "Конгломерат, брекчия", "Conglomerate, breccia",
      contains=("conglomerate", "breccia"), commons=("Conglomerate_(geology)",), indicates=("placer",), tests=()),
    C("concretion", "sedimentary_rock", "Конкреция", "Concretion", contains=("concretion", "septarian", "nodule"),
      commons=("Concretions",), indicates=(), tests=()),
    C("marble", "metamorphic_rock", "Мрамор", "Marble", contains=("marble",), commons=("Marble",), indicates=(),
      tests=("acid_fizz",)),
    C("gneiss_schist", "metamorphic_rock", "Гнейс, сланец кристаллический", "Gneiss, schist",
      contains=("gneiss", "schist", "migmatite", "amphibolite", "eclogite", "slate", "phyllite"),
      commons=("Gneiss", "Schist"), indicates=("metamorphic", "orogenic_gold"), tests=("foliated",)),
    C("sinter", "hydrothermal_rock", "Гейзерит, кремнистый туф", "Siliceous sinter",
      contains=("sinter", "geyserite"), commons=("Siliceous_sinter",), indicates=("epithermal",), tests=()),
    C("fulgurite", "hydrothermal_rock", "Фульгурит", "Fulgurite", contains=("fulgurite",), commons=("Fulgurites",),
      indicates=(), tests=()),
    C("meteorite", "meteorite", "Метеорит", "Meteorite", contains=("meteorite",), commons=("Meteorites",),
      indicates=("meteorite",), tests=("magnetic", "heavy", "fusion_crust")),
)

BY_KEY = {c.key: c for c in CLASSES}
assert len(BY_KEY) == len(CLASSES), "duplicate class key"
assert all(c.group in GROUPS for c in CLASSES)

_EXACT = {name: c.key for c in CLASSES for name in c.exact}
_CONTAINS = [(re.compile(r"\b" + re.escape(w)), c.key) for c in CLASSES for w in c.contains]
# Words that make a name useless as a label: containers, labels, sample prep, analysis
_NOT_A_SPECIMEN = re.compile(
    r"\b(box|bottle|tube|jar|slide|label|thin section|polished section|tooth|core|powder|cast|replica|model|board)\b")


def base_name(raw: str) -> str:
    """'Gold on Quartz (var. X), Colorado' -> 'gold'. Lower case, first mineral only."""
    t = (raw or "").lower()
    t = re.sub(r"\(.*?\)", " ", t)
    t = re.split(r",| with | on | in | and | & |;", t)[0]
    t = re.sub(r"\b(var|variety)\.?\b.*", "", t)
    t = re.sub(r"[^a-z\- ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def map_name(raw: str, topics: tuple[str, ...] | list[str] = ()) -> str | None:
    """Raw specimen name (+ museum topic) -> class key, or None if it is not a usable label."""
    if any("meteorite" in (t or "").lower() for t in topics):
        return "meteorite"
    low = (raw or "").lower()
    if not low or _NOT_A_SPECIMEN.search(low):
        return None
    b = base_name(raw)
    if b in _EXACT:
        return _EXACT[b]
    for rx, key in _CONTAINS:          # rocks: 'pink granite', 'porphyritic granite'
        if rx.search(b):
            return key
    return None


# What a class MAY indicate. Never a claim that a deposit exists.
INDICATORS = {
    "gold": {"ru": "Похоже на золото. Если это оно, рядом могут быть кварцевые жилы или россыпи. Подтвердить может только проба в лаборатории.",
             "en": "Looks like gold. If it is, quartz veins or placer gravels may be nearby. Only a lab assay can confirm it."},
    "gold_associate": {"ru": "Пирит часто встречается вместе с золотом в рудных жилах, но сам по себе золота не означает.",
                       "en": "Pyrite often occurs with gold in ore veins, but on its own it does not mean gold is present."},
    "silver": {"ru": "Минерал серебра: признак серебряной или полиметаллической минерализации.",
               "en": "A silver mineral: a sign of silver or base-metal mineralisation."},
    "copper_primary": {"ru": "Медная минерализация (первичная). Стоит проверить окрестности на зелёные и синие налёты.",
                       "en": "Copper mineralisation (primary). Worth checking the area for green and blue stains."},
    "copper_oxide_zone": {"ru": "Зелёные и синие медные минералы образуются у поверхности над медными рудами. Это классический поисковый признак.",
                          "en": "Green and blue copper minerals form near the surface above copper ores. A classic prospecting sign."},
    "lead_zinc": {"ru": "Признак свинцово-цинковой (часто с серебром) минерализации.",
                  "en": "A sign of lead-zinc mineralisation, often with silver."},
    "sulfide": {"ru": "Сульфидный минерал: указывает на рудные процессы, но встречается и в обычных породах.",
                "en": "A sulfide mineral: points to ore-forming processes, but also occurs in ordinary rocks."},
    "nickel": {"ru": "В основных породах пирротин бывает спутником никелевых руд.",
               "en": "In mafic rocks pyrrhotite can accompany nickel ores."},
    "iron": {"ru": "Железистый минерал. Скопления могут быть железной рудой.",
             "en": "An iron mineral. Large masses can be iron ore."},
    "gossan": {"ru": "Ржавые бурые корки (железная шляпа) иногда лежат над сульфидными рудами. Повод присмотреться, а не вывод.",
               "en": "Rusty brown crusts (gossan) sometimes cap sulfide ores. A reason to look closer, not a conclusion."},
    "tin": {"ru": "Касситерит: главный минерал олова, часто в гранитных районах и россыпях.",
            "en": "Cassiterite: the main tin mineral, often in granite areas and placers."},
    "quartz_vein": {"ru": "Кварц очень распространён. Кварцевые жилы в метаморфических породах иногда несут золото, но в большинстве его нет.",
                    "en": "Quartz is very common. Quartz veins in metamorphic rocks sometimes carry gold, but most do not."},
    "orogenic_gold": {"ru": "В таких породах бывают золотоносные кварцевые жилы (орогенный тип).",
                      "en": "Such rocks can host gold-bearing quartz veins (orogenic type)."},
    "epithermal": {"ru": "Следы древних горячих источников: в таких системах бывает золото и серебро (эпитермальный тип).",
                   "en": "Traces of ancient hot springs: such systems can carry gold and silver (epithermal type)."},
    "nickel_chrome_pgm": {"ru": "Ультраосновные породы: возможны хром, никель, платиноиды.",
                          "en": "Ultramafic rocks: chromium, nickel and platinum-group metals are possible."},
    "diamond": {"ru": "Алмазы связаны с кимберлитами и россыпями. По фото алмаз почти всегда путают с кварцем.",
                "en": "Diamonds come from kimberlites and placers. On photos diamond is almost always confused with quartz."},
    "pegmatite": {"ru": "Признак пегматитов: там бывают литий, тантал, драгоценные камни.",
                  "en": "A sign of pegmatites: they can carry lithium, tantalum, gemstones."},
    "gem": {"ru": "Драгоценный или поделочный камень. Ценность определяет геммолог.",
            "en": "A gem or ornamental stone. Only a gemmologist can value it."},
    "manganese": {"ru": "Минерал марганца.", "en": "A manganese mineral."},
    "hydrothermal": {"ru": "Образуется из горячих растворов, часто рядом с рудными жилами.",
                     "en": "Forms from hot fluids, often near ore veins."},
    "evaporite": {"ru": "Осадок испарившейся воды (соли, гипс).", "en": "Deposited by evaporating water (salts, gypsum)."},
    "phosphate": {"ru": "Фосфатный минерал (сырьё для удобрений в крупных залежах).",
                  "en": "A phosphate mineral (fertiliser raw material in large deposits)."},
    "volcanic": {"ru": "Вулканическое происхождение.", "en": "Volcanic origin."},
    "mafic": {"ru": "Тёмные основные породы и минералы.", "en": "Dark mafic rocks and minerals."},
    "metamorphic": {"ru": "Метаморфическое происхождение.", "en": "Metamorphic origin."},
    "placer": {"ru": "Обломочная порода: в древних галечниках бывают россыпи золота.",
               "en": "A clastic rock: old gravels can hold placer gold."},
    "oil_source": {"ru": "Тёмные глинистые сланцы бывают нефтематеринскими, но нефть по фото камня не определить.",
                   "en": "Dark shales can be oil source rocks, but oil cannot be detected from a photo of a rock."},
    "not_gold_flakes": {"ru": "Блёстки слюды часто принимают за золото. Слюда расщепляется на листочки, золото мнётся.",
                        "en": "Mica flakes are often mistaken for gold. Mica splits into sheets, gold bends."},
    "meteorite": {"ru": "Похоже на метеорит. Проверьте магнитом и сообщите в геологический музей: находка может быть ценной.",
                  "en": "May be a meteorite. Test with a magnet and contact a geology museum: the find may be valuable."},
}

TESTS = {
    "malleable": {"ru": "Мнётся иглой, а не крошится", "en": "Dents with a needle instead of crumbling"},
    "streak_gold": {"ru": "Черта на неглазурованной керамике золотисто-жёлтая", "en": "Streak on unglazed porcelain is golden yellow"},
    "streak_black": {"ru": "Черта на неглазурованной керамике чёрная или зеленовато-чёрная", "en": "Streak on unglazed porcelain is black or greenish black"},
    "streak_red": {"ru": "Черта красно-бурая", "en": "Streak is red-brown"},
    "streak_brown": {"ru": "Черта светло-бурая", "en": "Streak is light brown"},
    "streak_grey": {"ru": "Черта свинцово-серая", "en": "Streak is lead grey"},
    "streak_white": {"ru": "Черта белая или светлая", "en": "Streak is white or pale"},
    "heavy": {"ru": "Заметно тяжелее обычного камня того же размера", "en": "Noticeably heavier than an ordinary stone of the same size"},
    "brittle": {"ru": "Хрупкий, крошится", "en": "Brittle, crumbles"},
    "not_gold": {"ru": "Отличие от золота: твёрдый и хрупкий, золото мягкое и мнётся", "en": "Unlike gold: hard and brittle, gold is soft and bends"},
    "soft": {"ru": "Мягкий, царапается ножом", "en": "Soft, scratched by a knife"},
    "soft_knife": {"ru": "Царапается стальным ножом", "en": "Scratched by a steel knife"},
    "fingernail": {"ru": "Царапается ногтем", "en": "Scratched by a fingernail"},
    "smell_sulfur": {"ru": "Запах серы при трении", "en": "Smells of sulfur when rubbed"},
    "very_hard": {"ru": "Царапает стекло и кварц", "en": "Scratches glass and quartz"},
    "hard_scratches_glass": {"ru": "Царапает стекло", "en": "Scratches glass"},
    "no_cleavage": {"ru": "Скол раковистый, без ровных плоскостей", "en": "Breaks with curved surfaces, no flat cleavage"},
    "cubic_cleavage": {"ru": "Раскалывается на кубики", "en": "Breaks into cubes"},
    "two_cleavages": {"ru": "Две ровные плоскости раскола под прямым углом", "en": "Two flat cleavage planes at right angles"},
    "splits_in_sheets": {"ru": "Расщепляется на тонкие гибкие листочки", "en": "Splits into thin flexible sheets"},
    "magnetic": {"ru": "Притягивается магнитом", "en": "Attracted by a magnet"},
    "magnetic_weak": {"ru": "Слабо притягивается магнитом", "en": "Weakly attracted by a magnet"},
    "acid_fizz": {"ru": "Шипит от капли уксуса или слабой кислоты", "en": "Fizzes with a drop of vinegar or weak acid"},
    "crystals_visible": {"ru": "Видны отдельные кристаллы разного цвета", "en": "Separate crystals of different colours are visible"},
    "fine_grained_dark": {"ru": "Тёмная, мелкозернистая, кристаллы не видны глазом", "en": "Dark and fine-grained, crystals not visible to the eye"},
    "conchoidal": {"ru": "Раковистый стеклянный излом", "en": "Glassy curved fracture"},
    "gritty": {"ru": "Шершавый, видны песчинки", "en": "Gritty, sand grains visible"},
    "foliated": {"ru": "Полосчатая или слоистая текстура", "en": "Banded or layered texture"},
    "fusion_crust": {"ru": "Тонкая тёмная корочка оплавления снаружи", "en": "Thin dark melted crust on the outside"},
}

CAUTIONS = {
    "asbestos": {"ru": "Серпентиниты могут содержать асбест: не пилите и не дробите без защиты.",
                 "en": "Serpentinites can contain asbestos: do not saw or crush without protection."},
}

for _c in CLASSES:
    assert all(k in INDICATORS for k in _c.indicates), _c.key
    assert all(k in TESTS for k in _c.tests), _c.key
    assert all(k in CAUTIONS for k in _c.caution), _c.key
