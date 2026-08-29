"""Synthetic shopping / ecommerce corpus: reviews, listings, buyer Q&A, a spec sheet.

Everything here is FICTIONAL. Aurora, Halcyon and Northwind are invented brands;
the Aurora 14, Aurora 14 Pro, Aurora Dock, Halcyon Buds, Halcyon Buds Pro,
Northwind Router N600 and Northwind Router N600X do not exist. Vellum Market,
Kestrel Electronics, Brightline Direct and Tidewater Supply are invented
retailers, and Dana Whitlock, M. Okada, TrailRunner88, Priya Sandoval and Glen
Ashcroft are invented reviewers. Every price, promotion, warranty term and
firmware version is made up.

The corpus exists to stress two things an ecommerce graph gets wrong, which are
*different* problems that pull in opposite directions:

  * **Near-duplicate product surfaces (a precision problem).** ``Aurora 14`` and
    ``Aurora 14 Pro`` differ by one token; ``Northwind Router N600`` and
    ``Northwind Router N600X`` differ by one *character*, and ``N600`` is a
    literal substring of ``N600X``. Any resolver that blocks on normalised edit
    distance merges them. They are genuinely different SKUs -- different charger,
    port count, radio, warranty length, price -- and every document that names
    both spells the difference out, so a resolver that reads context has enough
    to keep them apart. ``DISTINCT_PAIRS`` is the answer key for *not* merging.
  * **Alias variation on the same product (a recall problem).** The same laptop
    is written ``Aurora 14``, ``Aurora-14``, ``Aurora14``, ``the Aurora 14
    laptop`` and ``the base model``; the same earbuds are ``Halcyon Buds``,
    ``the Buds``, ``Halcyon earbuds`` and ``the standard Buds``.
    ``ALIAS_GROUPS`` is the gold clustering, and every surface form in it occurs
    literally in some ``DOCUMENTS[i]["text"]``.

The third thing the corpus is for is **aspect-level opinion**. A review sentiment
label is close to useless here: almost every review praises one aspect and
attacks another in the same paragraph ("the screen is the best I have had ...
battery life is where the Aurora14 falls down"). What a shopping graph actually
needs is ``(product, aspect, polarity)``, which means the aspect has to be
extracted as a node and attached to the right product. ``GOLD_ASPECTS`` scores
that pass on its own, separately from the relation extraction.

Cross-corpus hazard, on purpose: ``Halcyon`` and ``Northwind`` are also company
names in :mod:`kgx.data.documents`, where they are a semiconductor firm and a
freight operator. Merging the consumer-electronics brand ``Northwind`` with
``Northwind Logistics Inc.`` because the strings match is exactly the failure a
name-only resolver makes and a context-aware one should not.

``GOLD_SHOPPING_FACTS`` is the hand-written answer key of triples a perfect
extractor + resolver should recover, expressed with canonical names.
"""

DOCUMENTS: list[dict] = [
    {
        "doc_id": "s01",
        "kind": "listing",
        "product_hint": "Aurora 14",
        "title": "Aurora 14 — 14-inch ultrabook | Brightline Direct",
        "text": (
            "Aurora 14 (2026 edition), sold and shipped by Brightline Direct. The Aurora 14 "
            "laptop is Aurora's mainstream ultrabook: a 14-inch 2880x1800 OLED display at "
            "120 Hz, 16 GB of memory, a 1 TB SSD and a 68 Wh battery in a 1.29 kg magnesium "
            "chassis.\n\n"
            "Listed at $1,299. Through 14 September the Back-to-Campus promotion takes $150 "
            "off at checkout. Shipping: the Aurora 14 ships with free two-day delivery on any "
            "order over $99. Brightline Direct does not offer in-store pickup. Every "
            "Aurora-14 includes a two-year limited warranty; the Aurora Care+ extended plan "
            "adds accidental damage cover for $129 and is honoured only by Brightline "
            "Direct.\n\n"
            "The Aurora Dock is sold separately and is not in this box.\n\n"
            "Please check the model before ordering. The Aurora 14 is not the same machine as "
            "the Aurora 14 Pro. The base model uses a 45 W charger and has two USB-C ports "
            "and integrated graphics; the Pro ships with a 65 W charger, four USB-C ports and "
            "a discrete GPU, and costs $250 more. Category: laptops > ultrabooks."
        ),
    },
    {
        "doc_id": "s02",
        "kind": "review",
        "product_hint": "Aurora 14",
        "title": "Great screen, disappointing battery — Aurora 14 after six weeks",
        "text": (
            "Dana Whitlock, verified buyer. I bought the Aurora-14 from Kestrel Electronics in "
            "July and have used it daily since.\n\n"
            "The screen is genuinely the best I have had at this price — colour is accurate "
            "out of the box and I have stopped carrying an external monitor. Build quality is "
            "excellent too: the magnesium lid barely flexes and the hinge is still tight after "
            "six weeks of being thrown in a bag.\n\n"
            "Battery life is where the Aurora14 falls down. Aurora advertises eighteen hours; "
            "with two browsers and a video call I get closer to six. Fan noise is the other "
            "irritation — the fans spin up loudly during any compile, and I have taken to "
            "working in another room.\n\n"
            "Shipping was excellent. Kestrel had it on my desk in two days at no charge. "
            "Customer support is another story: I opened a ticket about the fan curve on "
            "3 August and heard nothing for nine days.\n\n"
            "At $1,299 I still think the Aurora 14 laptop is worth it and I would recommend it "
            "to anyone who mostly writes and reads. If you compile all day, look at the "
            "Aurora 14 Pro instead."
        ),
    },
    {
        "doc_id": "s03",
        "kind": "review",
        "product_hint": "Aurora 14 Pro",
        "title": "Aurora 14 Pro — worth the extra $250, but only if you need the GPU",
        "text": (
            "M. Okada. Upgraded from the Aurora 14 to the Aurora 14 Pro in August.\n\n"
            "These are two different machines whatever the near-identical product photos on "
            "Vellum Market suggest. The Pro has the discrete GPU, four USB-C ports, a 75 Wh "
            "battery and a 65 W charger, and it is 180 g heavier than the base model.\n\n"
            "Performance is the reason to buy it. A render that took eleven minutes on the "
            "Aurora 14 finishes in four on the Aurora 14 Pro. Battery life is better in "
            "practice too despite the larger draw; I get nine hours of light work.\n\n"
            "The keyboard is a downgrade. Travel is shallower than on the base model and I "
            "have made more typos in a month than in a year on the old machine. Build quality "
            "is otherwise identical and still very good.\n\n"
            "Brightline Direct wanted $1,549. I paid $1,469 at Kestrel Electronics with an "
            "Autumn Days code, which is a fair price for what you get. The Aurora Dock works "
            "properly with the Pro — two 4K outputs at 60 Hz, which it will not do on the "
            "Aurora 14. Recommended, with the keyboard caveat."
        ),
    },
    {
        "doc_id": "s04",
        "kind": "qa",
        "product_hint": "Aurora Dock",
        "title": "Q&A: Does the Aurora Dock work with the standard Aurora 14?",
        "text": (
            "Q: Will the Aurora Dock drive two external monitors off the Aurora 14, or do I "
            "need the Pro?\n\n"
            "A (Brightline Direct support): The Aurora Dock is compatible with the Aurora 14 "
            "and with the Aurora 14 Pro, but those are not the same product and the Dock does "
            "not behave the same way on each. On the Aurora 14 Pro you get two 4K displays at "
            "60 Hz. On the Aurora 14 the second output is capped at 30 Hz, because the base "
            "model's USB-C ports carry fewer display lanes. The Dock is $249 and carries the "
            "same two-year limited warranty as the laptops.\n\n"
            "Q: Is it in the box?\n\n"
            "A: No, the Aurora Dock is sold separately. During Back-to-Campus we do bundle it "
            "with the Aurora 14 Pro for $199, and the Aurora 14 Pro ships with free two-day "
            "delivery from Brightline Direct.\n\n"
            "Q: Will the 45 W charger that came with my Aurora-14 charge through the Dock?\n\n"
            "A: It will charge, slowly. Use the 65 W travel charger for full speed."
        ),
    },
    {
        "doc_id": "s05",
        "kind": "spec_sheet",
        "product_hint": "Aurora 14",
        "title": "Spec sheet: Aurora 14, Aurora 14 Pro and Aurora Dock (2026 line-up)",
        "text": (
            "Aurora 14 — Manufacturer: Aurora. Category: ultrabooks. Display: 14-inch "
            "2880x1800 OLED, 120 Hz. Memory: 16 GB. Storage: 1 TB SSD. Battery: 68 Wh. "
            "Charger: 45 W USB-C. Ports: two USB-C, one headphone. Weight: 1.29 kg. Graphics: "
            "integrated. MSRP $1,299. Warranty: two-year limited warranty.\n\n"
            "Aurora 14 Pro — Manufacturer: Aurora. Category: ultrabooks. Display: 14-inch "
            "2880x1800 OLED, 120 Hz. Memory: 32 GB. Storage: 2 TB SSD. Battery: 75 Wh. "
            "Charger: 65 W USB-C. Ports: four USB-C, one HDMI, one headphone. Weight: "
            "1.47 kg. Graphics: discrete. MSRP $1,549. Warranty: two-year limited "
            "warranty.\n\n"
            "Aurora Dock — Manufacturer: Aurora. Category: docking stations. Compatible with "
            "the Aurora 14 and the Aurora 14 Pro. Two DisplayPort outputs, one HDMI, 100 W "
            "passthrough. MSRP $249. Warranty: two-year limited warranty.\n\n"
            "Note for catalogue teams: the Aurora 14 and the Aurora 14 Pro share a chassis but "
            "are separate SKUs. The Aurora 14 Pro does not replace the Aurora 14; both remain "
            "in the 2026 line-up and both are stocked by Brightline Direct."
        ),
    },
    {
        "doc_id": "s06",
        "kind": "listing",
        "product_hint": "Halcyon Buds",
        "title": "Halcyon Buds — true wireless earbuds | Vellum Market",
        "text": (
            "Halcyon Buds, made by Halcyon, sold on Vellum Market by Tidewater Supply. These "
            "are the standard Halcyon earbuds, not the Halcyon Buds Pro.\n\n"
            "IPX4 water resistance, 6 hours of playback per charge and 24 hours with the "
            "charging case, Bluetooth 5.4, and a 12-month manufacturer warranty. $89.99, or "
            "$76.49 during Vellum Autumn Days, the sale Vellum Market runs from 1 to "
            "9 October. Shipping: each pair of Halcyon Buds ships with free standard shipping "
            "on orders over $35, or with next-day courier for $6.99, quoted at checkout.\n\n"
            "The Halcyon Buds are bundled with three sizes of silicone ear tips and a short "
            "braided USB-C cable in the box. The Buds sit in the true wireless earbuds "
            "category and are configured "
            "through the Halcyon Companion app.\n\n"
            "Compatibility note: the silicone ear tips fit both the Halcyon Buds and the "
            "Halcyon Buds Pro, but the Halcyon charging case is model-specific. The case for "
            "the Buds Pro is 4 mm deeper and will not close on the standard Buds."
        ),
    },
    {
        "doc_id": "s07",
        "kind": "review",
        "product_hint": "Halcyon Buds",
        "title": "Halcyon Buds: the sound is fine, the left bud is not",
        "text": (
            "TrailRunner88. Two months with the Halcyon earbuds.\n\n"
            "Sound quality is better than the $89.99 price suggests. Bass is controlled rather "
            "than boomy and voices are clear on podcasts. Battery life matches the claim — "
            "about six hours, and the charging case tops them up twice before it needs a "
            "cable.\n\n"
            "The problem is the left bud. It drops out for a second or two every ten minutes "
            "on my commute, and a factory reset did not fix it. Two other people I know with "
            "the Buds report the same dropout, so I do not think mine is a one-off "
            "defect.\n\n"
            "Fit is the other issue. Even on the largest silicone ear tips the Halcyon Buds "
            "work loose when I run, which is what I bought them for.\n\n"
            "Tidewater shipped in three days, no complaints there. Halcyon support replied "
            "within a day and offered a replacement under the 12-month manufacturer warranty, "
            "which I will take. I would not recommend the Buds for running. At a desk they are "
            "good value."
        ),
    },
    {
        "doc_id": "s08",
        "kind": "review",
        "product_hint": "Halcyon Buds Pro",
        "title": "Halcyon Buds Pro — the noise cancellation is the whole product",
        "text": (
            "Priya Sandoval, verified buyer. I owned the standard Halcyon Buds for a year and "
            "bought the Halcyon Buds Pro in June from Kestrel Electronics at $189.\n\n"
            "Do not assume these are the same earbuds with a nicer badge. The Buds Pro has a "
            "different driver, a larger charging case and active noise cancellation that the "
            "standard Buds simply does not have.\n\n"
            "Noise cancellation is excellent. It takes an open-plan office down to a hum and "
            "on a flight it beats headphones costing twice as much. Comfort over a four-hour "
            "stretch is good too; the Pro buds sit deeper than the standard ones and I stopped "
            "noticing them.\n\n"
            "Price is hard to defend. $189 is a lot when the Halcyon Buds are $89.99 and sound "
            "nearly as good in a quiet room. Build quality worries me as well: the lid of the "
            "charging case already has play in it after two months, which is not what I expect "
            "at this price.\n\n"
            "Kestrel's free two-day delivery arrived early, and the 12-month manufacturer "
            "warranty is the same on both models."
        ),
    },
    {
        "doc_id": "s09",
        "kind": "qa",
        "product_hint": "Halcyon Buds Pro",
        "title": "Q&A: Halcyon Buds and Halcyon Buds Pro accessories",
        "text": (
            "Q: Do the silicone ear tips from my Halcyon Buds fit the Buds Pro?\n\n"
            "A (Tidewater Supply): Yes. Halcyon uses the same tip mount on both, so the "
            "three-size tip pack is compatible with the Halcyon Buds and with the Halcyon Buds "
            "Pro.\n\n"
            "Q: Can I use one charging case for both?\n\n"
            "A: No, and this catches people out. These are different products. The Halcyon "
            "charging case shipped with the Halcyon Buds Pro is 4 mm deeper and its lid will "
            "not close on the standard Buds; going the other way, the smaller case does not "
            "seat the Pro buds well enough to charge them reliably. Order the case for the "
            "model you actually own.\n\n"
            "Q: Is the Buds Pro a replacement for the Halcyon Buds?\n\n"
            "A: No. Both are current products. The Halcyon Buds Pro sits above the Buds in the "
            "range at $189 against $89.99, and Halcyon has not discontinued the Buds.\n\n"
            "Q: Warranty and returns?\n\n"
            "A: Both carry the 12-month manufacturer warranty. Tidewater Supply handles "
            "returns within 30 days; after that, contact Halcyon support directly."
        ),
    },
    {
        "doc_id": "s10",
        "kind": "listing",
        "product_hint": "Northwind Router N600",
        "title": "Northwind Router N600 — clearance | Kestrel Electronics",
        "text": (
            "Northwind Router N600, made by Northwind, on clearance at Kestrel Electronics for "
            "$59, down from $119. The Clearance Blowout runs until stock is gone.\n\n"
            "Read the model number before you order. This is the outgoing N600, superseded in "
            "March by the Northwind Router N600X, and the two are not interchangeable — the "
            "N600 router will not mesh with an N600X node.\n\n"
            "The Northwind N600 is a dual-band Wi-Fi 6 mesh router covering roughly 1,800 "
            "square feet per node, with four gigabit LAN ports and a 12-month manufacturer "
            "warranty. Category: networking > mesh routers. It is set up through the Northwind "
            "Home app.\n\n"
            "The N600 ships with free standard shipping, and Kestrel Electronics offers "
            "in-store pickup on it the same day.\n\n"
            "Accessory note: the N600X wall mount kit does not fit the N600. The mounting "
            "holes moved between the two models."
        ),
    },
    {
        "doc_id": "s11",
        "kind": "listing",
        "product_hint": "Northwind Router N600X",
        "title": "Northwind Router N600X — tri-band Wi-Fi 7 mesh | Vellum Market",
        "text": (
            "The Northwind Router N600X replaces the Northwind Router N600 in Northwind's mesh "
            "line-up.\n\n"
            "Tri-band Wi-Fi 7, roughly 2,400 square feet of coverage per node, one 2.5 GbE WAN "
            "port and three gigabit LAN ports, and a 24-month manufacturer warranty — double "
            "the cover on the older N600. $179 on Vellum Market, sold by Vellum Market "
            "directly. Shipping: the N600X ships with free two-day delivery for Vellum Market "
            "members. Vellum Autumn Days takes $20 off the N600X.\n\n"
            "In the two-pack, the N600X is bundled with a second N600X node and with the N600X "
            "wall mount kit, for $329. "
            "The N600X is in the mesh routers category and is configured through the Northwind "
            "Home app.\n\n"
            "Important: the N600X will not mesh with a Northwind N600. If you are extending an "
            "existing N600 network, buy another N600 rather than this. Northwind offers a "
            "trade-in credit of $25 against an old N600 when you buy the N600X."
        ),
    },
    {
        "doc_id": "s12",
        "kind": "review",
        "product_hint": "Northwind Router N600X",
        "title": "N600X review: coverage solved, the app is still miserable",
        "text": (
            "Glen Ashcroft, verified buyer. Replaced two ageing N600 nodes with a pair of the "
            "Northwind N600X in July.\n\n"
            "Coverage is the headline and the N600X delivers. The garden office that used to "
            "drop to one bar now sits at full signal, and range through a stone wall is far "
            "better than the N600 router ever managed. Setup took eleven minutes.\n\n"
            "After that, the Northwind Home app is miserable. It logs me out weekly, the "
            "channel settings are three menus deep, and firmware 2.1.4 bricked one of my nodes "
            "for a day until I found the reset pinhole. That firmware complaint is all over the "
            "forums, so it is not just me.\n\n"
            "Vellum Market delivered in two days and $179 is a fair price against what the "
            "competition charges. Northwind support was slow — four days to answer a firmware "
            "question, and the first reply was a copy-paste of the manual.\n\n"
            "Build quality is solid. I would recommend the N600X, but buy it for the radios, "
            "not the software."
        ),
    },
    {
        "doc_id": "s13",
        "kind": "qa",
        "product_hint": "Northwind Router N600",
        "title": "Q&A: N600 or N600X — which one do I need?",
        "text": (
            "Q: I already own a Northwind N600. Kestrel Electronics has the N600 at $59 and "
            "Vellum Market has the N600X at $179. Can I mix them?\n\n"
            "A: No. The Northwind Router N600 and the Northwind Router N600X are different "
            "products despite the one-character difference in the name. The N600 is dual-band "
            "Wi-Fi 6; the N600X is tri-band Wi-Fi 7. They do not form a single mesh, so adding "
            "an N600X to an N600 network leaves you with two separate networks.\n\n"
            "Q: So is the N600 dead?\n\n"
            "A: The Northwind Router N600X replaces it and Northwind stopped manufacturing the "
            "N600 in March, but Kestrel Electronics still has clearance stock and it is still "
            "covered by the 12-month manufacturer warranty.\n\n"
            "Q: Will my wall mount transfer?\n\n"
            "A: No. The N600X wall mount kit is compatible with the N600X only.\n\n"
            "Q: Which would you recommend?\n\n"
            "A: Starting fresh, the N600X. Adding a node to an existing N600 network, the "
            "N600."
        ),
    },
    {
        "doc_id": "s14",
        "kind": "review",
        "product_hint": "Aurora Dock",
        "title": "Kestrel Electronics order review: fast shipping, painful returns",
        "text": (
            "Dana Whitlock, verified buyer. This is about Kestrel Electronics rather than the "
            "hardware. I ordered an Aurora 14 and an Aurora Dock together on 12 July.\n\n"
            "Shipping was excellent: free two-day delivery, both boxes on my step inside 40 "
            "hours, and the laptop carton was undamaged.\n\n"
            "The Dock was not. It arrived with a crushed corner and a bent HDMI port, so the "
            "packaging clearly did not survive the courier. Kestrel's returns process is where "
            "the order fell apart. The online form rejected my order number three times, the "
            "phone queue ran to 35 minutes, and the replacement Aurora Dock took eleven days "
            "to arrive. Nobody would honour the Aurora Care+ cover I had paid $129 for at "
            "checkout, which I later learned only Brightline Direct supports.\n\n"
            "Prices are fine — $1,299 for the Aurora-14 and $249 for the Dock, the same as "
            "everywhere else. But I will buy the next laptop from Brightline Direct, and I "
            "would not recommend Kestrel Electronics for anything you might need to send back."
        ),
    },
]


# (head_canonical, relation, tail_canonical) triples a perfect pipeline should
# recover from DOCUMENTS. Relation vocabulary is closed; see kgx.domains.SHOPPING.
GOLD_SHOPPING_FACTS: list[tuple[str, str, str]] = [
    # --- manufacturer ------------------------------------------------------
    ("Aurora 14", "made_by", "Aurora"),
    ("Aurora 14 Pro", "made_by", "Aurora"),
    ("Aurora Dock", "made_by", "Aurora"),
    ("Halcyon Buds", "made_by", "Halcyon"),
    ("Halcyon Buds Pro", "made_by", "Halcyon"),
    ("Northwind Router N600", "made_by", "Northwind"),
    ("Northwind Router N600X", "made_by", "Northwind"),
    # --- channel -----------------------------------------------------------
    ("Aurora 14", "sold_by", "Brightline Direct"),
    ("Aurora 14", "sold_by", "Kestrel Electronics"),
    ("Aurora 14 Pro", "sold_by", "Kestrel Electronics"),
    ("Halcyon Buds", "sold_by", "Tidewater Supply"),
    ("Halcyon Buds Pro", "sold_by", "Kestrel Electronics"),
    ("Northwind Router N600", "sold_by", "Kestrel Electronics"),
    ("Northwind Router N600X", "sold_by", "Vellum Market"),
    # --- taxonomy ----------------------------------------------------------
    ("Aurora 14", "belongs_to_category", "ultrabooks"),
    ("Aurora Dock", "belongs_to_category", "docking stations"),
    ("Halcyon Buds", "belongs_to_category", "true wireless earbuds"),
    ("Northwind Router N600X", "belongs_to_category", "mesh routers"),
    # --- price -------------------------------------------------------------
    ("Aurora 14", "priced_at", "$1,299"),
    ("Aurora 14 Pro", "priced_at", "$1,549"),
    ("Aurora Dock", "priced_at", "$249"),
    ("Halcyon Buds", "priced_at", "$89.99"),
    ("Halcyon Buds Pro", "priced_at", "$189"),
    ("Northwind Router N600", "priced_at", "$59"),
    ("Northwind Router N600X", "priced_at", "$179"),
    # --- accessories and bundles -------------------------------------------
    ("Aurora Dock", "compatible_with", "Aurora 14 Pro"),
    ("Aurora Dock", "compatible_with", "Aurora 14"),
    ("silicone ear tips", "compatible_with", "Halcyon Buds Pro"),
    ("N600X wall mount kit", "compatible_with", "Northwind Router N600X"),
    ("Aurora 14 Pro", "bundled_with", "Aurora Dock"),
    ("Halcyon Buds", "bundled_with", "silicone ear tips"),
    ("Northwind Router N600X", "bundled_with", "N600X wall mount kit"),
    # --- lifecycle. The ONLY successor relation in the corpus; Aurora 14 Pro
    #     and Halcyon Buds Pro are explicitly NOT successors (s05, s09).
    ("Northwind Router N600X", "replaces", "Northwind Router N600"),
    # --- warranty, promotion, shipping -------------------------------------
    ("Aurora 14", "covered_by", "two-year limited warranty"),
    ("Halcyon Buds", "covered_by", "12-month manufacturer warranty"),
    ("Northwind Router N600X", "covered_by", "24-month manufacturer warranty"),
    ("Aurora 14", "discounted_by", "Back-to-Campus promotion"),
    ("Halcyon Buds", "discounted_by", "Vellum Autumn Days"),
    ("Northwind Router N600", "discounted_by", "Clearance Blowout"),
    ("Vellum Autumn Days", "offered_by", "Vellum Market"),
    ("Aurora Care+", "offered_by", "Brightline Direct"),
    ("Aurora 14", "ships_with", "free two-day delivery"),
    ("Halcyon Buds", "ships_with", "free standard shipping"),
    ("Northwind Router N600", "ships_with", "free standard shipping"),
    ("Northwind Router N600X", "ships_with", "free two-day delivery"),
    ("Kestrel Electronics", "ships_with", "in-store pickup"),
    # --- specifications ----------------------------------------------------
    ("Aurora 14", "has_specification", "68 Wh battery"),
    ("Aurora 14 Pro", "has_specification", "75 Wh battery"),
    ("Northwind Router N600X", "has_specification", "tri-band Wi-Fi 7"),
    ("Northwind Router N600", "has_specification", "dual-band Wi-Fi 6"),
    # --- comparison, opinion -----------------------------------------------
    ("Aurora 14 Pro", "compares_to", "Aurora 14"),
    ("Halcyon Buds Pro", "compares_to", "Halcyon Buds"),
    ("Northwind Router N600X", "compares_to", "Northwind Router N600"),
    ("Dana Whitlock", "recommends", "Aurora 14"),
    ("M. Okada", "recommends", "Aurora 14 Pro"),
    ("Glen Ashcroft", "recommends", "Northwind Router N600X"),
    ("Dana Whitlock", "complains_about", "Kestrel Electronics"),
    ("TrailRunner88", "complains_about", "Halcyon Buds"),
    # s08 is the aspect-level case: Priya Sandoval is positive about the Buds Pro
    # overall and negative about two specific facets, so the complaint attaches to
    # the aspect, not the product. A document-level sentiment label loses this.
    ("Priya Sandoval", "complains_about", "price"),
    ("Priya Sandoval", "complains_about", "build quality"),
]


# Aspect-level opinion, scored separately from the relation pass. A whole-review
# sentiment label would be wrong for almost every one of these documents: s02
# praises the screen and attacks the battery in adjacent paragraphs, s12 praises
# the radios and attacks the app. "mixed" is reserved for an aspect the SAME
# document argues both ways about, not for a document with mixed aspects.
GOLD_ASPECTS: list[dict] = [
    {"doc_id": "s02", "product": "Aurora 14", "aspect": "screen", "sentiment": "positive"},
    {"doc_id": "s02", "product": "Aurora 14", "aspect": "build quality", "sentiment": "positive"},
    {"doc_id": "s02", "product": "Aurora 14", "aspect": "battery life", "sentiment": "negative"},
    {"doc_id": "s02", "product": "Aurora 14", "aspect": "fan noise", "sentiment": "negative"},
    {"doc_id": "s02", "product": "Kestrel Electronics", "aspect": "shipping", "sentiment": "positive"},
    {"doc_id": "s02", "product": "Aurora 14", "aspect": "customer support", "sentiment": "negative"},
    {"doc_id": "s02", "product": "Aurora 14", "aspect": "price", "sentiment": "positive"},
    {"doc_id": "s03", "product": "Aurora 14 Pro", "aspect": "performance", "sentiment": "positive"},
    {"doc_id": "s03", "product": "Aurora 14 Pro", "aspect": "battery life", "sentiment": "positive"},
    {"doc_id": "s03", "product": "Aurora 14 Pro", "aspect": "keyboard", "sentiment": "negative"},
    {"doc_id": "s03", "product": "Aurora 14 Pro", "aspect": "price", "sentiment": "mixed"},
    {"doc_id": "s07", "product": "Halcyon Buds", "aspect": "sound quality", "sentiment": "positive"},
    {"doc_id": "s07", "product": "Halcyon Buds", "aspect": "battery life", "sentiment": "positive"},
    {"doc_id": "s07", "product": "Halcyon Buds", "aspect": "connectivity", "sentiment": "negative"},
    {"doc_id": "s07", "product": "Halcyon Buds", "aspect": "fit", "sentiment": "negative"},
    {"doc_id": "s07", "product": "Halcyon Buds", "aspect": "customer support", "sentiment": "positive"},
    {"doc_id": "s08", "product": "Halcyon Buds Pro", "aspect": "noise cancellation", "sentiment": "positive"},
    {"doc_id": "s08", "product": "Halcyon Buds Pro", "aspect": "comfort", "sentiment": "positive"},
    {"doc_id": "s08", "product": "Halcyon Buds Pro", "aspect": "price", "sentiment": "negative"},
    {"doc_id": "s08", "product": "Halcyon Buds Pro", "aspect": "build quality", "sentiment": "negative"},
    {"doc_id": "s12", "product": "Northwind Router N600X", "aspect": "coverage", "sentiment": "positive"},
    {"doc_id": "s12", "product": "Northwind Router N600X", "aspect": "setup", "sentiment": "positive"},
    {"doc_id": "s12", "product": "Northwind Router N600X", "aspect": "app", "sentiment": "negative"},
    {"doc_id": "s12", "product": "Northwind Router N600X", "aspect": "firmware", "sentiment": "negative"},
    {"doc_id": "s12", "product": "Northwind Router N600X", "aspect": "customer support", "sentiment": "negative"},
    {"doc_id": "s12", "product": "Northwind Router N600X", "aspect": "build quality", "sentiment": "positive"},
    {"doc_id": "s14", "product": "Kestrel Electronics", "aspect": "shipping", "sentiment": "positive"},
    {"doc_id": "s14", "product": "Aurora Dock", "aspect": "packaging", "sentiment": "negative"},
    {"doc_id": "s14", "product": "Kestrel Electronics", "aspect": "returns", "sentiment": "negative"},
]


# Gold clustering for entity resolution (B-cubed). canonical -> every surface
# form that literally occurs in some DOCUMENTS[i]["text"]. Verified by
# ``_verify_aliases()`` below.
ALIAS_GROUPS: dict[str, list[str]] = {
    "Aurora 14": [
        "Aurora 14",
        "Aurora-14",
        "Aurora14",
        "the Aurora 14 laptop",
        "the base model",
    ],
    "Aurora 14 Pro": [
        "Aurora 14 Pro",
        "the Aurora 14 Pro",
        "the Pro",
    ],
    "Aurora Dock": [
        "Aurora Dock",
        "the Aurora Dock",
        "the Dock",
    ],
    "Halcyon Buds": [
        "Halcyon Buds",
        "the Halcyon Buds",
        "the Buds",
        "Halcyon earbuds",
        "the standard Buds",
        "the standard Halcyon Buds",
    ],
    "Halcyon Buds Pro": [
        "Halcyon Buds Pro",
        "the Halcyon Buds Pro",
        "the Buds Pro",
        "the Pro buds",
    ],
    "Northwind Router N600": [
        "Northwind Router N600",
        "Northwind N600",
        "the N600",
        "the N600 router",
    ],
    "Northwind Router N600X": [
        "Northwind Router N600X",
        "Northwind N600X",
        "the N600X",
        "an N600X",
    ],
    "Aurora": [
        "Aurora",
        "Aurora's",
    ],
    "Halcyon": [
        "Halcyon",
    ],
    "Northwind": [
        "Northwind",
        "Northwind's",
    ],
    "Kestrel Electronics": [
        "Kestrel Electronics",
        "Kestrel",
        "Kestrel's",
    ],
    "Vellum Market": [
        "Vellum Market",
    ],
    "Brightline Direct": [
        "Brightline Direct",
    ],
    "Tidewater Supply": [
        "Tidewater Supply",
        "Tidewater",
    ],
    "Dana Whitlock": [
        "Dana Whitlock",
    ],
}


# Pairs that a normalised-edit-distance or substring blocker will happily merge
# and that are DIFFERENT products. This is the precision answer key: a resolver
# is scored on how many of these it keeps apart, not only on how many aliases it
# pulls together. ("N600" is a literal substring of "N600X"; "Aurora 14" is a
# literal prefix of "Aurora 14 Pro"; the brand name is a prefix of every product.)
DISTINCT_PAIRS: list[tuple[str, str]] = [
    ("Aurora 14", "Aurora 14 Pro"),
    ("Northwind Router N600", "Northwind Router N600X"),
    ("Halcyon Buds", "Halcyon Buds Pro"),
    ("Aurora 14", "Aurora Dock"),
    ("Aurora", "Aurora 14"),
    ("Halcyon", "Halcyon Buds"),
    ("Northwind", "Northwind Router N600"),
    ("two-year limited warranty", "12-month manufacturer warranty"),
    ("12-month manufacturer warranty", "24-month manufacturer warranty"),
    ("Aurora Care+", "two-year limited warranty"),
    ("free two-day delivery", "free standard shipping"),
    ("silicone ear tips", "N600X wall mount kit"),
]


def _verify_aliases() -> dict[str, list[str]]:
    """Surface forms in ``ALIAS_GROUPS`` that do NOT occur in the corpus.

    Run as ``python -m kgx.data.shopping``. An empty dict means the gold
    clustering is honest about the text; a non-empty one means a scoring run
    would be measuring a form the extractor can never produce.
    """
    blob = "\n".join(d["text"] for d in DOCUMENTS)
    missing: dict[str, list[str]] = {}
    for canon, forms in ALIAS_GROUPS.items():
        absent = [f for f in forms if f not in blob]
        if absent:
            missing[canon] = absent
    return missing


if __name__ == "__main__":  # pragma: no cover
    words = [len(d["text"].split()) for d in DOCUMENTS]
    print(f"{len(DOCUMENTS)} documents, {min(words)}-{max(words)} words")
    print(f"{len(GOLD_SHOPPING_FACTS)} gold facts, {len(GOLD_ASPECTS)} gold aspects")
    print("missing aliases:", _verify_aliases() or "none")
