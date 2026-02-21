"""
Sustainability Technology Pre-Filter v2.2

This module defines a pre-filter for evaluating articles related to sustainability technology.
Fast keyword-based filtering before model inference.

v2.2 Changes:
- Enhanced product deals detection (Jackery case study)
- Added: "Green Deals" column detection, "$X savings/discount", urgency language
- Added: "exclusive/limited + price", "new low price", "starting from $X"

v2.1 Changes:
- Added product deals exclusion (shopping/price content)
- Added trade show patterns (CES, IFA, MWC) with sustainability override
- Expanded consumer electronics brands and product types
- Expanded sustainability override keywords

v2.0 Changes:
- Added explicit exclusion patterns for off-topic content
- Blocks AI/ML infrastructure, consumer electronics, programming tutorials, etc.
- Added multi-lingual inclusion keywords (20+ languages)
"""

import re
from typing import Dict, List, Optional, Tuple
from filters.common.base_prefilter import BasePreFilter


class SustainabilityTechnologyPreFilterV2(BasePreFilter):
    """
    Pre-filter to evaluate articles on sustainability technology.
    v2.1: Enhanced exclusions + expanded override protection.
    """
    VERSION = "2.2"

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
    }

    # Sustainability keywords that override exclusions (context-dependent)
    # If article contains these, it passes even if exclusion patterns match
    SUSTAINABILITY_OVERRIDE = [
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
        self.filter_name = "sustainability_technology_v2"
        self.version = "2.2"

    def apply_filter(self, article: Dict) -> Tuple[bool, str]:
        """
        Determine if article should be sent to LLM for scoring.

        Returns:
            (should_score, reason)
            - (True, "passed"): Send to LLM
            - (False, reason): Block from LLM
        """
        # NOTE: Commerce prefilter integration point TBD
        # Decision postponed until model performance is validated

        text = self._get_combined_clean_text(article)
        title = article.get('title', '').lower()

        # FIRST: Check exclusions (blocks even if sustainability keywords present)
        excluded, exclusion_reason = self._is_excluded(title, text)
        if excluded:
            return (False, exclusion_reason)

        # SECOND: Check sustainability relevance
        if not self._is_sustainability_related(text):
            return (False, "not_sustainability_topic")

        # PASS: Has sustainability evidence AND passed all other checks
        return (True, "passed")

    def _is_excluded(self, title: str, text: str) -> Tuple[bool, str]:
        """
        Check if article matches exclusion patterns.
        Returns (excluded, reason).
        """
        combined = f"{title} {text[:1000]}".lower()

        for category, patterns in self.EXCLUSION_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, combined, re.IGNORECASE):
                    # Check if sustainability keywords override the exclusion
                    if any(kw in combined for kw in self.SUSTAINABILITY_OVERRIDE):
                        # Has sustainability context, don't exclude
                        continue
                    return (True, f"excluded_{category}")

        return (False, "")

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
