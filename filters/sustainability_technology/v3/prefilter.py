"""
Sustainability Technology Pre-Filter v3.0

ADR-018 declarative shape: subclass declares EXCLUSION_PATTERNS,
OVERRIDE_KEYWORDS, and an optional _filter_specific_final_check() hook;
BasePreFilter.apply_filter() drives the standard pipeline.

History:
- v3.0 (2026-04-28): migrated to declarative BasePreFilter shape (#52,
  ADR-018). No behavior change vs v2.2 — pattern set, override keywords,
  and "is sustainability-related" check are identical. Class name stays
  as ...V2 until the version-drift batch rename (also part of #52).
- v2.2: enhanced product deals detection (Jackery case study).
- v2.1: product deals exclusion, trade show patterns with sustainability
  override, expanded brands/product types and override keywords.
- v2.0: explicit exclusion patterns for off-topic content; multi-lingual
  inclusion keywords (20+ languages).
"""

from typing import Dict, List, Tuple
from filters.common.base_prefilter import BasePreFilter


class SustainabilityTechnologyPreFilterV2(BasePreFilter):
    """
    Pre-filter to evaluate articles on sustainability technology.
    v3.0 (declarative shape per ADR-018).
    """
    VERSION = "3.0"

    # Exclusion patterns - these block articles before sustainability check
    # NOTE: Articles with SUSTAINABILITY_OVERRIDE keywords bypass these exclusions
    # PHILOSOPHY: When in doubt, let it through - the LLM oracle handles edge cases
    EXCLUSION_PATTERNS = {
        # AI/ML infrastructure without sustainability application
        # REMOVED: transformer (electrical transformers), benchmark (ESG benchmarks)
        'ai_ml_infrastructure': [
            r'\b(attention mechanism|SOTA|state.of.the.art)\b',
            r'\b(diffusion model|GAN|VAE|autoencoder|neural network architecture)\b',
            r'\b(LLM|language model|GPT-\d|BERT|Llama|Claude|Gemini|Mistral)\b',
            r'\b(fine.?tun|pretrain|inference optimization|token generation)\b',
            r'\b(image classification|object detection|segmentation|computer vision)\b',
        ],
        # Consumer electronics reviews/shopping
        # REMOVED: Dyson (makes sustainable products), GPU (data center efficiency)
        'consumer_electronics': [
            # Smartphones
            r'\b(Galaxy S\d|iPhone \d|Pixel \d|OnePlus|Xiaomi Mi|Oppo Find|Vivo X|Redmi)\b',
            r'\b(smartphone review|tablet review|phone deal|best phone)\b',
            # Gaming hardware (specific)
            r'\b(RTX \d{4}|gaming laptop|gaming PC|PlayStation|Xbox)\b',
            # Audio equipment
            r'\b(soundbar|earbuds|headphones|wireless speaker|subwoofer|AirPods)\b',
            # TV/Display (non-efficiency focused)
            r'\b(OLED TV|Mini.?LED TV|QD.?OLED|TV review|television review)\b',
            # Brands (removed Dyson)
            r'\b(TCL|JLab|Anker|Narwal|Roborock|iRobot|Bose|Sony WH-|Sonos)\b',
        ],
        # Product deals/shopping - only obvious shopping content
        # REMOVED: "lowest price", "best deal" (could be sustainability milestones)
        # REMOVED: "buying guide" (could be "best EVs to buy")
        'product_deals': [
            r'\b(Black Friday|Prime Day|Cyber Monday|holiday deal)\b',
            r'\b(discount code|coupon code|promo code)\b',
            r'\b(save \$\d+|percent off|\d+% off)\b',
            r'\bgift guide\b',
            # v2.2: Enhanced deal detection (Jackery case)
            r'\bGreen Deals\b',                                      # Electrek's deals column
            r'\$\d[\d,]*\s*(savings|discount|off)\b',                # "$700 savings", "$500 off"
            r'\bdeals?\b.{0,30}(ending|expire|tonight|today only)',  # "deals ending tonight"
            r'\b(exclusive|limited).{0,15}(low|price|deal|offer)\b', # "exclusive new lows"
            r'\b(new|all.time|record)\s+low.{0,10}(price|from|\$)',  # "new low price"
            r'\bstarting\s+(at|from)\s+\$\d',                        # "starting from $1,219"
        ],
        # Trade shows - only block gadget-focused coverage
        'trade_shows': [
            r'\b(hands.on|first look).{0,20}(CES|IFA|MWC)\b',
            r'\b(CES|IFA|MWC).{0,20}(hands.on|first look)\b',
        ],
        # Home appliances - only obviously non-sustainable
        # REMOVED: smart refrigerator/fridge (energy efficiency!)
        'home_appliances': [
            r'\b(mattress vacuum|air fryer|instant pot|coffee maker|espresso machine)\b',
        ],
        # Programming/developer content
        # REMOVED: tutorial, how to build (DIY sustainability is valid)
        'programming': [
            r'\b(REST API|GraphQL|microservice)\b',
            r'\bgithub\.com/(?!.*(?:solar|energy|carbon|climate|sustainab))\b',
        ],
        # Military technology
        # REMOVED: naval (offshore wind, naval renewable energy)
        'military': [
            r'\b(fighter jet|missile|warship|stealth bomber|defense system)\b',
            r'\b(military weapon|armament|air force base)\b',
            r'\b(battle tank|army tank|military tank)\b',
        ],
        # Travel/tourism
        # REMOVED: airline (sustainable aviation fuel is important!)
        'travel': [
            r'\b(travel app|flight deal|vacation package|tourism promotion)\b',
            r'\b(hotel booking|trip planning|world cup trip)\b',
        ],
        # Clickbait / listicle framing (#46).
        # TODO(#51-equivalent): if a universal clickbait detector is built (per
        # the obituary-detector pattern in issue #51), these regexes will move
        # to a shared module with per-filter consumption policy. Until then,
        # per-filter scope. SUSTAINABILITY_OVERRIDE keywords still pass through
        # — legitimate solar/wind articles with clickbait phrasing are NOT
        # blocked here (e.g. "You Won't Believe What Solar Panels Are Made Of"
        # has "solar" in the override list and falls through).
        # Skipped issue items 7-8 (anti-example overfits on the 22-Carat Gold
        # body) — items 1-6 catch the same article via the title alone.
        'clickbait': [
            # Canonical clickbait phrasing. The .? on contractions matches
            # "won't" / "wont" / "don't" / "dont".
            r'\b(you won.?t believe|doctors hate|this one weird|one simple trick)\b',
            # "Throws away without knowing" — the 22-Carat Gold pattern.
            r'\bwithout (knowing|realizing|even trying)\b',
            # Listicle framing: "this common electronic item", "that common mistake".
            r'\b(this|that) common .{0,40}(item|mistake|thing|food|habit)\b',
            # "You're probably/likely doing/making/missing X"
            r"\byou.?re (probably|likely) (doing|making|missing)\b",
            # "10 things you didn't know" — both halves required, with a
            # proximity cap (.{0,120}) so a legit "7 Ways Solar Homeowners Are
            # Cutting Bills" article with a colloquial "you didn't expect..."
            # later in the body doesn't trip the unbounded .* (review finding,
            # would have leaked without the bound).
            r"\b\d+\s+(things|reasons|ways|tricks|secrets|facts)\b.{0,120}\byou (don.?t|didn.?t|never)\b",
            # "Shocking truth", "mind-blowing fact", etc.
            r'\b(shocking|amazing|mind.?blowing|jaw.?dropping) (fact|truth|discovery)\b',
        ],
    }

    # Sustainability keywords that override exclusions (context-dependent).
    # If article contains any of these substrings (case-insensitive), exclusion
    # patterns are bypassed. Consumed by BasePreFilter._has_override (ADR-018).
    OVERRIDE_KEYWORDS = [
        # Carbon & emissions
        'carbon reduction', 'carbon footprint', 'carbon neutral', 'co2 reduction',
        'emission reduction', 'emissions reduction', 'reduce emission', 'ghg',
        'decarboni', 'net zero', 'net-zero',
        # Energy efficiency
        'energy efficien', 'energy-efficien', 'energy saving', 'energy management',
        # Renewables - solar
        'renewable', 'solar power', 'solar panel', 'solar energy', 'photovoltaic',
        # Renewables - wind (including offshore)
        'wind power', 'wind energy', 'wind farm', 'wind turbine', 'offshore wind',
        # Climate
        'climate change', 'climate action', 'green energy', 'clean energy',
        # Electric transport (multiple patterns)
        'electric vehicle', 'electric car', 'electric sedan', 'electric suv',
        ' ev ', ' evs ', 'ev4', 'ev6', 'ev9',  # Specific EV models
        'ev charging', 'zero emission', 'e-bike', 'ebike',
        # Storage & grid
        'battery storage', 'energy storage', 'grid storage', 'smart grid',
        # Sustainability general
        'sustainab', 'circular economy', 'carbon capture',
    ]

    def __init__(self):
        """Initialize the sustainability technology prefilter."""
        super().__init__()
        self.filter_name = "sustainability_technology_v3"
        self.version = "3.0"

    def _filter_specific_final_check(self, title: str, text: str) -> Tuple[bool, str]:
        """
        After exclusions pass, require at least one sustainability keyword
        anywhere in title+content. Without this, off-topic articles that don't
        match any exclusion pattern would slip through (e.g. local sports news).
        """
        if self._is_sustainability_related(text):
            return (True, "")
        return (False, "not_sustainability_topic")

    def _is_sustainability_related(self, text: str) -> bool:
        """Check if article is about climate/sustainability/clean energy (WIDEST NET)"""

        keywords = [
            # === ENGLISH ===
            # 1. Climate Core & Mitigation
            'climate', 'carbon', 'emission', 'greenhouse', 'warming',
            'net-zero', 'net zero', 'carbon neutral', 'decarboniz',
            'carbon capture', 'direct air capture', 'co2',
            'paris agreement', 'cop', 'unfccc', 'deforestation',

            # 2. Energy Transition & Materials (Technical)
            'renewable', 'solar', 'wind', 'geothermal', 'hydro', 'nuclear',
            'battery', 'energy storage', 'grid storage', 'hydrogen', 'electrif',
            'electric vehicle', 'electric car', 'electric truck', 'electric pickup',
            ' ev ', ' evs', 'bev', 'phev', 'tesla', 'rivian', 'lucid',
            'fossil fuel', 'coal', 'oil', 'gas',
            'critical material', 'lithium', 'copper', 'steel', 'plastic', 'mining',

            # 3. Resource Management & Circularity
            'sustainab', 'green energy', 'clean energy', 'circular economy',
            'recycl', 'waste reduction', 'water conservation', 'desalination',

            # 4. Land Use & Agriculture
            'regenerative', 'agrivoltaic', 'vertical farm', 'sustainable food',
            'biodiversity', 'reforestation',

            # 5. Policy, Finance & Systemic Context (The Wider Net)
            'esg', 'green bond', 'disclosure', 'sustainability report',
            'just transition', 'labor rights', 'indigenous', 'human rights',
            'infrastructure', 'innovation', 'regulations', 'mandate',
            'investments', 'subsidies', 'resilience', 'adaptation',
            'climate risk', 'stranded asset', 'supply chain',

            # === DUTCH (NL) ===
            'klimaat', 'koolstof', 'uitstoot', 'broeikasgas', 'opwarming',
            'duurzaam', 'hernieuwbaar', 'zonne-energie', 'zonnepane', 'windenergie',
            'elektrisch', 'elektrische auto', 'waterstof', 'batterij',
            'kernenergie', 'kerncentrale', 'circulair', 'recycl',
            'energie-transitie', 'energietransitie', 'groene energie',

            # === GERMAN (DE) ===
            'klima', 'kohlenstoff', 'treibhausgas', 'erwärmung', 'emissionen',
            'nachhaltig', 'erneuerbar', 'solarenergie', 'windkraft', 'wasserstoff',
            'elektroauto', 'elektrofahrzeug', 'elektromobil', 'batterie',
            'kernkraft', 'atomkraft', 'kreislaufwirtschaft', 'energiewende',
            'photovoltaik', 'wärmepumpe', 'grüne energie',

            # === FRENCH (FR) ===
            'climat', 'carbone', 'émission', 'gaz à effet de serre', 'réchauffement',
            'durable', 'renouvelable', 'solaire', 'éolien', 'hydrogène',
            'véhicule électrique', 'voiture électrique', 'batterie',
            'nucléaire', 'économie circulaire', 'transition énergétique',
            'photovoltaïque', 'énergie verte', 'décarbonation',

            # === SPANISH (ES) ===
            'clima', 'carbono', 'emisión', 'efecto invernadero', 'calentamiento',
            'sostenible', 'renovable', 'solar', 'eólica', 'hidrógeno',
            'vehículo eléctrico', 'coche eléctrico', 'batería',
            'nuclear', 'economía circular', 'transición energética',
            'fotovoltaica', 'energía verde', 'descarbonización',

            # === PORTUGUESE (PT) ===
            'clima', 'carbono', 'emissão', 'efeito estufa', 'aquecimento',
            'sustentável', 'renovável', 'solar', 'eólica', 'hidrogénio', 'hidrogênio',
            'veículo elétrico', 'carro elétrico', 'bateria',
            'nuclear', 'economia circular', 'transição energética',
            'fotovoltaica', 'energia verde', 'descarbonização',

            # === ITALIAN (IT) ===
            'clima', 'carbonio', 'emissione', 'effetto serra', 'riscaldamento',
            'sostenibile', 'rinnovabile', 'solare', 'eolico', 'idrogeno',
            'veicolo elettrico', 'auto elettrica', 'batteria',
            'nucleare', 'economia circolare', 'transizione energetica',
            'fotovoltaico', 'energia verde', 'decarbonizzazione',

            # === CHINESE (ZH) ===
            '气候', '碳', '排放', '温室气体', '变暖',
            '可持续', '可再生', '太阳能', '风能', '氢能',
            '电动汽车', '电动车', '电池', '新能源',
            '核能', '循环经济', '能源转型',
            '光伏', '绿色能源', '脱碳',

            # === SWEDISH (SV) ===
            'klimat', 'koldioxid', 'utsläpp', 'växthusgas', 'uppvärmning',
            'hållbar', 'förnybar', 'solenergi', 'vindkraft', 'vätgas',
            'elbil', 'elfordon', 'batteri', 'kärnkraft',
            'cirkulär ekonomi', 'energiomställning', 'grön energi',

            # === DANISH (DA) ===
            'klima', 'kulstof', 'udledning', 'drivhusgas', 'opvarmning',
            'bæredygtig', 'vedvarende', 'solenergi', 'vindenergi', 'brint',
            'elbil', 'batteri', 'atomkraft',
            'cirkulær økonomi', 'energiomstilling', 'grøn energi',

            # === NORWEGIAN (NO) ===
            'klima', 'karbon', 'utslipp', 'drivhusgass', 'oppvarming',
            'bærekraftig', 'fornybar', 'solenergi', 'vindkraft', 'hydrogen',
            'elbil', 'elektrisk kjøretøy', 'batteri', 'kjernekraft',
            'sirkulær økonomi', 'energiomstilling', 'grønn energi',

            # === FINNISH (FI) ===
            'ilmasto', 'hiili', 'päästö', 'kasvihuonekaasu', 'lämpeneminen',
            'kestävä', 'uusiutuva', 'aurinkoenergia', 'tuulivoima', 'vety',
            'sähköauto', 'akku', 'ydinvoima',
            'kiertotalous', 'energiasiirtymä', 'vihreä energia',

            # === POLISH (PL) ===
            'klimat', 'węgiel', 'emisja', 'gaz cieplarniany', 'ocieplenie',
            'zrównoważony', 'odnawialny', 'energia słoneczna', 'wiatrowa', 'wodór',
            'samochód elektryczny', 'pojazd elektryczny', 'bateria', 'jądrowy',
            'gospodarka obiegu zamkniętego', 'transformacja energetyczna', 'zielona energia',

            # === CZECH (CS) ===
            'klima', 'uhlík', 'emise', 'skleníkový plyn', 'oteplování',
            'udržitelný', 'obnovitelný', 'solární energie', 'větrná energie', 'vodík',
            'elektromobil', 'elektrické vozidlo', 'baterie', 'jaderný',
            'oběhové hospodářství', 'energetická transformace', 'zelená energie',

            # === RUSSIAN (RU) ===
            'климат', 'углерод', 'выброс', 'парниковый газ', 'потепление',
            'устойчивый', 'возобновляемый', 'солнечная энергия', 'ветровая', 'водород',
            'электромобиль', 'электрический автомобиль', 'аккумулятор', 'ядерный',
            'циркулярная экономика', 'энергетический переход', 'зелёная энергия',

            # === UKRAINIAN (UK) ===
            'клімат', 'вуглець', 'викид', 'парниковий газ', 'потепління',
            'сталий', 'відновлюваний', 'сонячна енергія', 'вітрова', 'водень',
            'електромобіль', 'електричний автомобіль', 'акумулятор', 'ядерний',
            'циркулярна економіка', 'енергетичний перехід', 'зелена енергія',

            # === GREEK (EL) ===
            'κλίμα', 'άνθρακας', 'εκπομπή', 'θερμοκήπιο', 'θέρμανση',
            'βιώσιμος', 'ανανεώσιμος', 'ηλιακή ενέργεια', 'αιολική', 'υδρογόνο',
            'ηλεκτρικό αυτοκίνητο', 'μπαταρία', 'πυρηνικός',
            'κυκλική οικονομία', 'ενεργειακή μετάβαση', 'πράσινη ενέργεια',

            # === HUNGARIAN (HU) ===
            'klíma', 'szén', 'kibocsátás', 'üvegházhatású gáz', 'felmelegedés',
            'fenntartható', 'megújuló', 'napenergia', 'szélenergia', 'hidrogén',
            'elektromos autó', 'elektromos jármű', 'akkumulátor', 'nukleáris',
            'körforgásos gazdaság', 'energiaátmenet', 'zöld energia',

            # === ROMANIAN (RO) ===
            'climă', 'carbon', 'emisie', 'gaz cu efect de seră', 'încălzire',
            'durabil', 'regenerabil', 'energie solară', 'eoliană', 'hidrogen',
            'mașină electrică', 'vehicul electric', 'baterie', 'nuclear',
            'economie circulară', 'tranziție energetică', 'energie verde',

            # === TURKISH (TR) ===
            'iklim', 'karbon', 'emisyon', 'sera gazı', 'ısınma',
            'sürdürülebilir', 'yenilenebilir', 'güneş enerjisi', 'rüzgar', 'hidrojen',
            'elektrikli araç', 'elektrikli otomobil', 'batarya', 'nükleer',
            'döngüsel ekonomi', 'enerji dönüşümü', 'yeşil enerji',

            # === ARABIC (AR) ===
            'مناخ', 'كربون', 'انبعاثات', 'غازات الدفيئة', 'احترار',
            'مستدام', 'متجدد', 'طاقة شمسية', 'طاقة الرياح', 'هيدروجين',
            'سيارة كهربائية', 'مركبة كهربائية', 'بطارية', 'نووي',
            'اقتصاد دائري', 'تحول الطاقة', 'طاقة خضراء',
        ]

        # Very permissive - just needs ONE keyword mention
        return any(kw in text for kw in keywords)


def test_prefilter():
    """Lightweight self-test for the prefilter. Mirror of belonging/v1/prefilter.py
    style — both will be unified once prefilter shape is harmonized (separate issue
    filed alongside #46)."""

    prefilter = SustainabilityTechnologyPreFilterV2()

    test_cases = [
        # Should BLOCK - Clickbait additions from issue #46 (NexusMind#157, #162)
        # Each one previously bypassed: pure clickbait with no sustainability
        # keyword, listicle "X Things You Didn't Know" framing, "One Weird Trick".
        {
            'title': 'People Throw Away This Common Electronic Item Without Knowing It Holds 450MG of Pure 22-Carat Gold',
            'text': 'Many household items contain hidden gold, with some everyday electronic devices holding up to '
                    '450 milligrams of pure 22-carat gold inside. Most people are unaware of the precious metals '
                    'inside their old gadgets and simply throw them away. Refurbishers and recyclers can extract '
                    'these materials, but the average consumer rarely thinks about the value sitting in a junk drawer. '
                    'Experts recommend taking devices to specialist recyclers rather than landfill.',
            'expected': (False, 'excluded_clickbait')
        },
        {
            'title': '10 Things You Didn\'t Know About Your Phone Battery',
            'text': 'Smartphone batteries have evolved enormously since the first cellular handsets, but most users '
                    'have only a passing understanding of how they work. Lithium-ion chemistry remains the dominant '
                    'choice. Charging habits affect long-term capacity. Heat is the enemy of battery longevity. '
                    'Modern phones include software safeguards to extend battery life. Manufacturers offer trade-in '
                    'programs for old devices. Battery replacement is now a common service at most repair shops.',
            'expected': (False, 'excluded_clickbait')
        },
        {
            'title': 'This One Weird Trick Saves Energy',
            'text': 'Households across the country are looking for ways to reduce their utility bills as energy '
                    'prices climb. A single behavioural change recommended by efficiency experts can cut consumption '
                    'noticeably without any equipment purchase. The trick involves rethinking how appliances are '
                    'used during peak hours. The savings add up over a billing cycle. Some utilities offer additional '
                    'rebates for similar behaviour. The approach has been studied for several years.',
            'expected': (False, 'excluded_clickbait')
        },

        # Should PASS - Legitimate sustainability content with rhetorical clickbait
        # phrasing. SUSTAINABILITY_OVERRIDE keyword "solar panel" / "solar" lets it
        # through. This is intentional per issue #46 design discussion.
        {
            'title': 'You Won\'t Believe What Solar Panels Are Actually Made Of',
            'text': 'Solar panels are everywhere now, but few people stop to think about the materials inside them. '
                    'The silicon cells, glass layers, and metal frames each have their own production story. The '
                    'aluminum frames typically come from a separate manufacturing chain than the photovoltaic cells '
                    'themselves. Recycling old solar panels is becoming a growing industry as the first generation '
                    'of installations reaches end of life. Manufacturers are also exploring ways to use less rare earths.',
            'expected': (True, 'passed')
        },

        # Should PASS - Pattern 5 cross-sentence non-regression. The "X ways" /
        # "you didn't" listicle pattern is bounded to .{0,120} so a distant
        # colloquial "you didn't" later in the body doesn't trip clickbait. Pre-
        # fix (with unbounded .*), this would have hit clickbait. Override on
        # "solar panel" / "battery storage" also saves it, but cleaner not to
        # rely on that. Review battery surfaced this case.
        {
            'title': '7 Ways Solar Homeowners Are Cutting Bills',
            'text': 'A growing number of homeowners with solar panel installations are finding new approaches to '
                    'maximise their savings. Industry analysts have published seven distinct strategies that solar '
                    'households can adopt this season. The strategies cover battery storage, time-of-use shifting, '
                    'panel orientation, and inverter upgrades among other techniques. Real installation case studies '
                    'illustrate each approach. Many years after the initial install, you didn\'t expect the payback '
                    'period to shrink so dramatically — but new tariff structures have made it real.',
            'expected': (True, 'passed')
        },

        # Should PASS - "common misconceptions" pattern would not match the listicle
        # CLICKBAIT regex (no "this/that common Y item|mistake|thing|food|habit" form),
        # AND has carbon capture anchor for sustainability check.
        {
            'title': 'Common Misconceptions About Carbon Capture',
            'text': 'Carbon capture and storage technology is the subject of significant debate, with both supporters '
                    'and critics often citing inaccurate facts. This article addresses several persistent misconceptions '
                    'about the energy cost of capture, the geological storage capacity, and the timeframe required for '
                    'commercial deployment. The technology has been operational at industrial scale in several locations '
                    'for over a decade. Recent advances have lowered the energy penalty considerably.',
            'expected': (True, 'passed')
        },
    ]

    print("Testing Sustainability Technology Pre-Filter v3.0")
    print("=" * 60)

    passed = 0
    failed = 0

    for i, test in enumerate(test_cases, 1):
        result = prefilter.apply_filter(test)
        expected = test['expected']
        result_matches = (
            result[0] == expected[0] and
            (result[1] == expected[1] or result[1].startswith(expected[1]))
        )
        status = "[PASS]" if result_matches else "[FAIL]"
        if result_matches:
            passed += 1
        else:
            failed += 1
        print(f"\nTest {i}: {status}")
        print(f"Title: {test['title'][:70]}...")
        print(f"Expected: {expected}")
        print(f"Got:      {result}")

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")


if __name__ == "__main__":
    test_prefilter()
