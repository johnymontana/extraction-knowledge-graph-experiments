"""Synthetic business/financial news corpus for the knowledge-graph extraction demo.

Everything in this module is FICTIONAL. Northwind Logistics Inc., Cascade Freight
Systems, Meridian Rail Group, Halcyon Semiconductor Corporation, Torrent
Microsystems and Vantage Energy Partners do not exist; neither do Priya Raman,
Marcus Webb, Dr. Elena Vasquez, Ana Duarte or any other person named here. The
tickers (NASDAQ: NWL, NYSE: HLCN), the dollar figures, the litigation captions
and the regulatory proceedings are all invented. Real agencies (the SEC, the
Surface Transportation Board, the European Commission) appear only as generic
counterparties in fabricated matters; nothing here describes any real action by
any real body.

The corpus is ten documents dated 2026-01-14 through 2026-07-02, written to be
*hard* in specific, checkable ways:

  * Entity name variants. "Northwind Logistics Inc." / "Northwind Logistics" /
    "Northwind" / "NWL" / "the Company"; "Halcyon Semiconductor Corporation" /
    "Halcyon Semiconductor" / "Halcyon" / "HLCN"; "Cascade Freight Systems" /
    "Cascade Freight" / "Cascade"; "Dr. Elena Vasquez" / "Vasquez";
    "Marcus Webb" / "Mr. Webb"; "Ana Duarte" / "Duarte". ``ALIAS_GROUPS`` is the
    gold clustering for scoring entity resolution (B-cubed).
  * Modality traps. Several sentences have the surface shape of an asserted fact
    but are hypothetical, negated, forward-looking or conditioned on approval:
    "The proposed acquisition, if approved ... would add ...", "The Company does
    not expect to acquire ...", "Management has no plans to divest ...", "We may
    face supply disruptions if tariffs ... rise", "Should demand ... soften,
    margins ... could compress". A pipeline without a modality/negation
    qualifier pass will happily assert all of them. See ``MODALITY_TRAPS``.
  * Cross-corpus hazard. Priya Raman, Marcus Webb, Dr. Elena Vasquez and
    Northwind Logistics are shared with the conversation corpus in
    ``kgx.data.conversations`` ON PURPOSE. Linking the two graphs on the company
    is useful; naively merging "Priya Raman, Chief Executive Officer" from a
    press release with "Priya Raman, the user in the chat log" is exactly the
    kind of over-merge the demo should surface rather than hide.

``GOLD_DOC_FACTS`` is the hand-written answer key: the triples a perfect
extractor + resolver should recover, expressed with canonical entity names.
"""

DOCUMENTS: list[dict] = [
    {
        "doc_id": "d01",
        "date": "2026-01-14",
        "source": "Press Release",
        "title": "Northwind Logistics to Acquire Cascade Freight Systems in $1.4 Billion Transaction",
        "text": (
            "SEATTLE — Northwind Logistics Inc. (NASDAQ: NWL) today announced that it has "
            "entered into a definitive agreement to acquire Cascade Freight Systems for "
            "$1.4 billion in cash and stock. Under the terms of the agreement, Cascade "
            "Freight shareholders will receive $28.50 per share, a 22% premium to the "
            "thirty-day volume weighted average price.\n\n"
            "Cascade Freight Systems operates 41 cross-dock terminals across the Pacific "
            "Northwest and reported revenue of $612 million in fiscal 2025. Following "
            "closing, Cascade will operate as a wholly owned subsidiary of Northwind "
            "Logistics and will retain its brand in the regional less-than-truckload "
            "market.\n\n"
            "\"Cascade gives us density in corridors where we have been renting capacity "
            "for a decade,\" said Priya Raman, Chief Executive Officer of Northwind "
            "Logistics. \"The proposed acquisition, if approved by the Surface "
            "Transportation Board, would add roughly 9,000 lane-miles of contracted volume "
            "to our network.\"\n\n"
            "Marcus Webb, Chief Financial Officer, said the Company expects the transaction "
            "to be accretive to adjusted earnings per share within eighteen months of "
            "closing and to generate $85 million of annual run-rate synergies by 2028. The "
            "Company does not expect to acquire additional terminal assets in 2026.\n\n"
            "The transaction has been unanimously approved by the boards of both companies "
            "and is expected to close in the second quarter of 2026, subject to regulatory "
            "clearance and customary closing conditions. Northwind Logistics has secured "
            "$900 million of committed bridge financing."
        ),
    },
    {
        "doc_id": "d02",
        "date": "2026-02-03",
        "source": "Reuters",
        "title": "US regulator opens review of Northwind-Cascade freight deal as Meridian pact expands",
        "text": (
            "The Surface Transportation Board said on Tuesday it had opened a formal review "
            "of Northwind's proposed purchase of Cascade, setting a 180-day clock that "
            "shipper groups had lobbied for.\n\n"
            "Northwind Logistics, the Seattle-based trucking and freight brokerage "
            "operator, said it would cooperate fully with the review. Shares of NWL closed "
            "1.8% lower.\n\n"
            "Separately, Northwind said it had expanded its intermodal partnership with "
            "Meridian Rail Group, adding six interchange points in Chicago, Memphis and "
            "Kansas City. Meridian, which operates 5,200 route-miles east of the "
            "Mississippi, has worked with the trucking company since 2019 on drayage "
            "handoffs. Northwind also disclosed that it acquired a 19.9% equity stake in "
            "Meridian Rail Group for $240 million, a position it described as strategic "
            "rather than a prelude to control.\n\n"
            "\"Management has no plans to divest the Cascade brand, or to fold its terminals "
            "into Northwind's existing network before 2028,\" Chief Executive Priya Raman "
            "told reporters on a call.\n\n"
            "Two shipper coalitions have asked regulators to impose service commitments as a "
            "condition of approval. Analysts at three brokerages said they still expected "
            "clearance, though one flagged that a longer review could push the closing into "
            "the fourth quarter.\n\n"
            "Northwind operates in the United States, Canada and the European Union, where "
            "it runs a smaller cross-border brokerage business. Reuters was unable to "
            "independently verify the synergy estimates the Company published in January."
        ),
    },
    {
        "doc_id": "d03",
        "date": "2026-02-19",
        "source": "8-K Excerpt",
        "title": "Northwind Logistics Inc. Form 8-K — Item 5.02 Director Election and Officer Retirement; Item 1A Risk Factor Update",
        "text": (
            "Item 5.02. On February 17, 2026, the Board of Directors of Northwind Logistics "
            "Inc. (the \"Company\") elected Dr. Elena Vasquez as a director, effective "
            "immediately, and appointed her chair of the Audit Committee. Dr. Elena Vasquez "
            "previously served as vice president of network science at a national parcel "
            "carrier and holds a doctorate in operations research. The Board has determined "
            "that Vasquez is independent under applicable Nasdaq listing standards.\n\n"
            "Also on February 17, 2026, Thomas Ingersoll notified the Company of his "
            "decision to retire as Chief Operating Officer effective March 31, 2026. Marcus "
            "Webb will assume interim oversight of the Telematics unit until a successor is "
            "named. Mr. Webb will continue to serve as Chief Financial Officer.\n\n"
            "Item 1A. The Company is supplementing its previously disclosed risk factors as "
            "follows. We may face supply disruptions if tariffs on imported semiconductor "
            "components rise, because a single vendor, Torrent Microsystems, supplies the "
            "controller boards used in substantially all of our telematics hardware. We do "
            "not currently expect to qualify a second source before 2027. Should demand for "
            "expedited freight soften, margins in our Freight Brokerage segment could "
            "compress meaningfully.\n\n"
            "The statements in this Item 1A are forward-looking and are not statements of "
            "historical fact. The Company undertakes no obligation to update them except as "
            "required by law."
        ),
    },
    {
        "doc_id": "d04",
        "date": "2026-03-05",
        "source": "Earnings Call Transcript",
        "title": "Northwind Logistics Inc. Fourth Quarter and Full Year 2025 Earnings Call",
        "text": (
            "Operator: Good morning and welcome to the Northwind Logistics fourth quarter "
            "and full year 2025 earnings call. All participants are in listen-only mode.\n\n"
            "Priya Raman, Chief Executive Officer: Thank you. Northwind delivered full year "
            "revenue of $2.41 billion, up 6.4% year over year, with fourth quarter revenue "
            "of $631 million. Adjusted operating margin for the year was 8.7%, up seventy "
            "basis points, and free cash flow was $187 million.\n\n"
            "Our Freight Brokerage segment produced $1.52 billion of revenue, our Dedicated "
            "Fleet segment $618 million, and our Telematics unit $274 million. Telematics "
            "grew 19% and is now our highest-margin business. By geography, the United "
            "States accounted for 78% of revenue, Canada 13%, and the European Union 9%. "
            "European volumes were roughly flat.\n\n"
            "During the quarter we signed a multi-year agreement to manage inbound logistics "
            "for Halcyon Semiconductor Corporation at its Phoenix fabrication campus, and we "
            "renewed our intermodal partnership with Meridian Rail Group.\n\n"
            "Marcus Webb, Chief Financial Officer: Thank you, Priya. Adjusted earnings per "
            "share were $3.12 for the year against $2.84 in 2024. Net debt to adjusted "
            "EBITDA finished the year at 1.9 times, and we expect that to rise to "
            "approximately 3.1 times at the close of the Cascade Freight Systems "
            "transaction.\n\n"
            "For 2026 we are guiding to revenue of $2.62 billion to $2.71 billion and "
            "adjusted operating margin of 8.4% to 9.0%. That guidance excludes Cascade. "
            "Diesel is our largest variable cost, and our supply contract with Vantage "
            "Energy Partners hedges roughly 60% of expected gallons through September.\n\n"
            "Analyst, Redwood Capital Markets: On the telematics supply chain — how exposed "
            "are you to Torrent Microsystems?\n\n"
            "Mr. Webb: Torrent supplies the controller boards for our gateway units. We "
            "qualify components about eighteen months ahead, so a near-term disruption would "
            "not change the 2026 plan. If tariffs on Asian-fabricated controllers were "
            "imposed at the rates now under discussion, our hardware cost of goods could "
            "rise by three to four points.\n\n"
            "Priya Raman: And to be direct about the speculation — we are not planning to "
            "exit the hardware business."
        ),
    },
    {
        "doc_id": "d05",
        "date": "2026-03-26",
        "source": "Press Release",
        "title": "Halcyon Semiconductor Names Ana Duarte Chief Executive Officer",
        "text": (
            "PHOENIX — Halcyon Semiconductor Corporation (NYSE: HLCN) today announced that "
            "its Board of Directors has appointed Ana Duarte as Chief Executive Officer, "
            "effective April 6, 2026. Duarte succeeds Robert Iyer, who is retiring after "
            "nine years leading the chipmaker.\n\n"
            "Duarte joined Halcyon in 2019 as president of the Automotive and Industrial "
            "group, where she oversaw the ramp of the company's 12-inch Phoenix fabrication "
            "campus. Under her leadership the group grew revenue from $410 million to $1.18 "
            "billion. The group produces power management controllers, radar front-ends and "
            "automotive-grade microcontrollers.\n\n"
            "\"Halcyon has the strongest analog roadmap in its history,\" Duarte said. \"Our "
            "priority is converting design wins into wafer starts, particularly in the "
            "segments where Torrent Microsystems has been most aggressive on price.\"\n\n"
            "Halcyon Semiconductor reported fiscal 2025 revenue of $4.86 billion and gross "
            "margin of 46.2%. The company operates fabrication facilities in Phoenix, "
            "Arizona and Dresden, Germany, and a test and assembly site in Penang, "
            "Malaysia.\n\n"
            "The Board also confirmed that HLCN will maintain its quarterly dividend of "
            "$0.24 per share. The company said it has no plans to divest the Dresden "
            "facility, notwithstanding recent press reports to the contrary. Halcyon's "
            "inbound logistics at the Phoenix campus are managed by Northwind Logistics "
            "under an agreement signed in December 2025."
        ),
    },
    {
        "doc_id": "d06",
        "date": "2026-04-15",
        "source": "Reuters",
        "title": "EU opens antitrust inquiry into Halcyon as price war with Torrent intensifies",
        "text": (
            "BRUSSELS — The European Commission has opened a formal antitrust inquiry into "
            "Halcyon Semiconductor over rebate agreements the chipmaker offered to European "
            "automotive customers, two people familiar with the matter said on "
            "Wednesday.\n\n"
            "The inquiry examines whether Halcyon conditioned volume rebates on customers "
            "sourcing at least 80% of their power management chips from the chipmaker, a "
            "practice rivals say foreclosed competition. Halcyon said it is cooperating and "
            "that its agreements comply with European law.\n\n"
            "The move lands in the middle of an unusually public fight with Torrent "
            "Microsystems, which has cut prices on comparable radar front-ends by an "
            "estimated 15% since November. Torrent competes with Halcyon across automotive "
            "analog and also supplies controller boards to the telematics unit of Northwind "
            "Logistics, the Seattle freight operator.\n\n"
            "Shares of HLCN fell 4.3% in New York. Ana Duarte, who became chief executive "
            "this month, told employees in a memo that the inquiry \"will not change our "
            "pricing discipline.\" Duarte has acknowledged that the chipmaker may face fines "
            "if the Commission ultimately finds against it, though no finding has been made "
            "and no statement of objections has been issued.\n\n"
            "Halcyon operates in Germany through its Dresden fabrication site, which "
            "supplies roughly a third of its European volume. Analysts said an adverse "
            "outcome, were one to come, would most likely arrive in 2028."
        ),
    },
    {
        "doc_id": "d07",
        "date": "2026-05-07",
        "source": "8-K Excerpt",
        "title": "Northwind Logistics Inc. Form 8-K — Item 8.01 Other Events",
        "text": (
            "Item 8.01. On May 4, 2026, Northwind Logistics Inc. (the \"Company\") received a "
            "subpoena from the Securities and Exchange Commission requesting documents "
            "relating to revenue recognition practices at Cascade Freight Systems for the "
            "periods from January 2023 through December 2025, including accruals for "
            "unbilled linehaul revenue.\n\n"
            "The subpoena was issued in connection with a formal order of investigation. The "
            "Company is cooperating with the SEC and has produced an initial document set. "
            "The Company cannot predict the outcome of the investigation, and an adverse "
            "resolution could result in monetary penalties or a restatement of the acquired "
            "entity's historical financial statements.\n\n"
            "Separately, on May 1, 2026, a putative securities class action captioned "
            "Delgado v. Northwind Logistics Inc. was filed in the United States District "
            "Court for the Western District of Washington. The complaint names the Company, "
            "Priya Raman and Marcus Webb as defendants and alleges that statements regarding "
            "the expected synergies of the Cascade Freight transaction were materially "
            "misleading. The Company believes the claims are without merit and intends to "
            "defend the action vigorously.\n\n"
            "The Surface Transportation Board review of the Cascade transaction remains "
            "pending. The Company does not expect the SEC inquiry to delay that review, "
            "although it may do so if additional document requests are issued."
        ),
    },
    {
        "doc_id": "d08",
        "date": "2026-05-21",
        "source": "Press Release",
        "title": "Vantage Energy Partners Signs Three-Year Fuel and Power Agreement with Northwind Logistics",
        "text": (
            "HOUSTON — Vantage Energy Partners today announced a three-year fuel supply and "
            "power agreement with Northwind Logistics Inc. covering approximately 190 "
            "million gallons of diesel annually across 34 terminals in the United States and "
            "Canada.\n\n"
            "Under the agreement, Vantage Energy Partners will supply diesel and renewable "
            "diesel blends to Northwind's dedicated fleet operations, with pricing indexed "
            "to a regional rack average plus a fixed differential. The agreement replaces a "
            "contract that expires in September 2026 and includes an option, exercisable by "
            "Northwind, to extend coverage to terminals acquired in the Cascade Freight "
            "transaction.\n\n"
            "Vantage also supplies firm electric capacity to Halcyon Semiconductor's Phoenix "
            "fabrication campus under a separate 2024 agreement, and operates renewable "
            "generation assets in Texas, Arizona and Alberta.\n\n"
            "\"Fuel is the single largest controllable cost in over-the-road freight, and "
            "diesel spreads have widened for three consecutive quarters,\" said Dana "
            "Okonkwo, chief commercial officer of Vantage Energy Partners.\n\n"
            "Marcus Webb, Chief Financial Officer of Northwind Logistics, said the agreement "
            "hedges roughly 60% of expected consumption through 2027. Mr. Webb noted that a "
            "sustained ten-cent move in the diesel rack price changes the annual fuel "
            "expense of Northwind Logistics by approximately $19 million. Neither party "
            "disclosed the total contract value."
        ),
    },
    {
        "doc_id": "d09",
        "date": "2026-06-11",
        "source": "Reuters",
        "title": "Northwind cuts 2026 guidance as diesel costs bite; shares slide 11%",
        "text": (
            "SEATTLE — Northwind Logistics cut its full-year revenue and margin guidance on "
            "Thursday, citing a 21% rise in diesel prices since March and softer spot demand "
            "in its brokerage business. Shares of NWL fell 11.4%, their steepest one-day "
            "decline since 2022.\n\n"
            "The Company now expects 2026 revenue of $2.48 billion to $2.55 billion, down "
            "from a February range of $2.62 billion to $2.71 billion, and adjusted operating "
            "margin of 7.1% to 7.6%, down from 8.4% to 9.0%.\n\n"
            "Chief Financial Officer Marcus Webb said the supply agreement with Vantage "
            "Energy Partners covered about 60% of gallons, leaving the balance exposed to "
            "spot rack prices. \"The unhedged portion cost us roughly $31 million in the "
            "quarter,\" Mr. Webb said on a call with analysts.\n\n"
            "Priya Raman said the Cascade Freight Systems acquisition remains on track and "
            "that the Surface Transportation Board review is expected to conclude this "
            "summer. Ms. Raman added that Northwind is not contemplating a change to its "
            "dividend.\n\n"
            "Two analysts downgraded NWL to hold. One noted that if diesel remains above "
            "$4.40 a gallon into the fourth quarter, the revised margin floor of the Company "
            "could prove optimistic. Telematics revenue, which is less fuel-sensitive, rose "
            "17% in the quarter, helped by hardware shipments built on controller boards "
            "from Torrent Microsystems."
        ),
    },
    {
        "doc_id": "d10",
        "date": "2026-07-02",
        "source": "Earnings Call Transcript",
        "title": "Halcyon Semiconductor Corporation Second Quarter 2026 Earnings Call",
        "text": (
            "Ana Duarte, Chief Executive Officer: Good afternoon. Halcyon delivered second "
            "quarter revenue of $1.29 billion, up 8.1% year over year, with non-GAAP gross "
            "margin of 47.4% and non-GAAP earnings per share of $1.36.\n\n"
            "Automotive and Industrial revenue was $702 million; Data Center and Networking "
            "was $421 million; the remainder came from our legacy discrete portfolio. By "
            "geography, the Americas represented 44% of revenue, Europe 31% and Asia Pacific "
            "25%. Our Dresden facility ran at 91% utilization; Phoenix ran at 84% as we "
            "bring the third module online.\n\n"
            "We continue to compete aggressively with Torrent Microsystems in radar "
            "front-ends, and we held share in the quarter despite their pricing. Our inbound "
            "logistics partner, Northwind Logistics, took over yard management at Phoenix in "
            "April and has cut average dwell time by nineteen hours.\n\n"
            "Analyst, Ironbridge Research: Any update on Brussels?\n\n"
            "Duarte: The European Commission inquiry is ongoing and we are cooperating. We "
            "do not expect a decision this fiscal year, and if the Commission were to issue "
            "a statement of objections, we would respond within the standard deadline.\n\n"
            "Chief Financial Officer of Halcyon Semiconductor: Capital expenditures were "
            "$310 million and we ended the quarter with $2.1 billion of cash. Our power "
            "agreement with Vantage Energy Partners insulates Phoenix from Arizona summer "
            "peak pricing. We are guiding third quarter revenue to $1.31 billion to $1.37 "
            "billion. Should Asian demand soften, that range could move lower."
        ),
    },
]


# (head_canonical, relation, tail_canonical) triples a perfect pipeline should
# recover from DOCUMENTS. Relation vocabulary is closed; see the demo ontology.
GOLD_DOC_FACTS: list[tuple[str, str, str]] = [
    # --- people / officers -------------------------------------------------
    ("Priya Raman", "officer_of", "Northwind Logistics Inc."),
    ("Marcus Webb", "officer_of", "Northwind Logistics Inc."),
    ("Dr. Elena Vasquez", "officer_of", "Northwind Logistics Inc."),
    ("Ana Duarte", "officer_of", "Halcyon Semiconductor Corporation"),
    # --- corporate structure ----------------------------------------------
    ("Northwind Logistics Inc.", "acquires", "Cascade Freight Systems"),
    ("Cascade Freight Systems", "subsidiary_of", "Northwind Logistics Inc."),
    ("Northwind Logistics Inc.", "has_stake_in", "Meridian Rail Group"),
    # --- commercial relationships -----------------------------------------
    ("Northwind Logistics Inc.", "partners_with", "Meridian Rail Group"),
    ("Northwind Logistics Inc.", "partners_with", "Halcyon Semiconductor Corporation"),
    ("Northwind Logistics Inc.", "supplies", "Halcyon Semiconductor Corporation"),
    ("Torrent Microsystems", "supplies", "Northwind Logistics Inc."),
    ("Vantage Energy Partners", "supplies", "Northwind Logistics Inc."),
    ("Vantage Energy Partners", "supplies", "Halcyon Semiconductor Corporation"),
    ("Halcyon Semiconductor Corporation", "competes_with", "Torrent Microsystems"),
    ("Torrent Microsystems", "competes_with", "Halcyon Semiconductor Corporation"),
    # --- geography ---------------------------------------------------------
    ("Northwind Logistics Inc.", "operates_in", "Seattle"),
    ("Northwind Logistics Inc.", "operates_in", "United States"),
    ("Northwind Logistics Inc.", "operates_in", "Canada"),
    ("Northwind Logistics Inc.", "operates_in", "European Union"),
    ("Cascade Freight Systems", "operates_in", "Pacific Northwest"),
    ("Halcyon Semiconductor Corporation", "operates_in", "Phoenix"),
    ("Halcyon Semiconductor Corporation", "operates_in", "Dresden"),
    ("Halcyon Semiconductor Corporation", "operates_in", "Penang"),
    # --- products ----------------------------------------------------------
    ("Halcyon Semiconductor Corporation", "produces", "power management controllers"),
    ("Halcyon Semiconductor Corporation", "produces", "radar front-ends"),
    ("Halcyon Semiconductor Corporation", "produces", "automotive-grade microcontrollers"),
    ("Torrent Microsystems", "produces", "controller boards"),
    # --- reported metrics --------------------------------------------------
    ("Northwind Logistics Inc.", "reports_metric", "FY2025 revenue of $2.41 billion"),
    ("Northwind Logistics Inc.", "reports_metric", "FY2025 adjusted operating margin of 8.7%"),
    ("Northwind Logistics Inc.", "reports_metric", "FY2025 adjusted EPS of $3.12"),
    ("Cascade Freight Systems", "reports_metric", "FY2025 revenue of $612 million"),
    ("Halcyon Semiconductor Corporation", "reports_metric", "FY2025 revenue of $4.86 billion"),
    ("Halcyon Semiconductor Corporation", "reports_metric", "Q2 2026 revenue of $1.29 billion"),
    # --- risk, legal, regulatory ------------------------------------------
    ("Northwind Logistics Inc.", "faces_risk", "diesel price increases"),
    ("Northwind Logistics Inc.", "faces_risk", "single-source semiconductor supply"),
    ("Halcyon Semiconductor Corporation", "faces_risk", "European antitrust fines"),
    ("Northwind Logistics Inc.", "party_to", "Delgado v. Northwind Logistics Inc."),
    ("Priya Raman", "party_to", "Delgado v. Northwind Logistics Inc."),
    ("Marcus Webb", "party_to", "Delgado v. Northwind Logistics Inc."),
    ("Northwind Logistics Inc.", "subject_to", "Securities and Exchange Commission"),
    ("Northwind Logistics Inc.", "subject_to", "Surface Transportation Board"),
    ("Halcyon Semiconductor Corporation", "subject_to", "European Commission"),
    # --- events ------------------------------------------------------------
    ("Priya Raman", "participant_in", "Northwind Logistics Q4 2025 earnings call"),
    ("Marcus Webb", "participant_in", "Northwind Logistics Q4 2025 earnings call"),
    ("Ana Duarte", "participant_in", "Halcyon Semiconductor Q2 2026 earnings call"),
    # --- causal ------------------------------------------------------------
    ("diesel prices", "impacts", "Northwind Logistics Inc."),
    ("Vantage Energy Partners", "impacts", "Northwind Logistics Inc."),
]


# Gold clustering for entity resolution (B-cubed). canonical -> every surface
# form that literally occurs in some DOCUMENTS[i]["text"].
ALIAS_GROUPS: dict[str, list[str]] = {
    "Northwind Logistics Inc.": [
        "Northwind Logistics Inc.",
        "Northwind Logistics",
        "Northwind",
        "Northwind's",
        "NWL",
        "the Company",
    ],
    "Cascade Freight Systems": [
        "Cascade Freight Systems",
        "Cascade Freight",
        "Cascade",
    ],
    "Meridian Rail Group": [
        "Meridian Rail Group",
        "Meridian",
    ],
    "Halcyon Semiconductor Corporation": [
        "Halcyon Semiconductor Corporation",
        "Halcyon Semiconductor",
        "Halcyon Semiconductor's",
        "Halcyon's",
        "Halcyon",
        "HLCN",
    ],
    "Torrent Microsystems": [
        "Torrent Microsystems",
        "Torrent",
    ],
    "Vantage Energy Partners": [
        "Vantage Energy Partners",
        "Vantage",
    ],
    "Priya Raman": [
        "Priya Raman",
        "Ms. Raman",
        "Priya",
    ],
    "Marcus Webb": [
        "Marcus Webb",
        "Mr. Webb",
        "Webb",
    ],
    "Dr. Elena Vasquez": [
        "Dr. Elena Vasquez",
        "Elena Vasquez",
        "Vasquez",
    ],
    "Ana Duarte": [
        "Ana Duarte",
        "Duarte",
    ],
    "Surface Transportation Board": [
        "Surface Transportation Board",
    ],
    "Securities and Exchange Commission": [
        "Securities and Exchange Commission",
        "SEC",
    ],
    "European Commission": [
        "European Commission",
        "the Commission",
    ],
}


# Sentences with the surface shape of an assertion that must NOT be stored as a
# plain asserted fact. A modality/negation qualifier pass should catch these.
MODALITY_TRAPS: list[dict] = [
    {
        "doc_id": "d01",
        "sentence": (
            "The proposed acquisition, if approved by the Surface Transportation Board, "
            "would add roughly 9,000 lane-miles of contracted volume to our network."
        ),
        "why": "hypothetical - conditional on regulatory approval, and forward-looking",
    },
    {
        "doc_id": "d01",
        "sentence": "The Company does not expect to acquire additional terminal assets in 2026.",
        "why": "negated expectation - no acquisition relation should be asserted",
    },
    {
        "doc_id": "d02",
        "sentence": (
            "Management has no plans to divest the Cascade brand, or to fold its terminals "
            "into Northwind's existing network before 2028,"
        ),
        "why": "negated intent - a divestiture is denied, not announced",
    },
    {
        "doc_id": "d03",
        "sentence": (
            "We may face supply disruptions if tariffs on imported semiconductor components "
            "rise, because a single vendor, Torrent Microsystems, supplies the controller "
            "boards used in substantially all of our telematics hardware."
        ),
        "why": (
            "mixed - the supplies(Torrent, Northwind) clause IS asserted, but the "
            "disruption is hypothetical and conditional on tariffs rising"
        ),
    },
    {
        "doc_id": "d03",
        "sentence": (
            "Should demand for expedited freight soften, margins in our Freight Brokerage "
            "segment could compress meaningfully."
        ),
        "why": "hypothetical - inverted conditional; no margin compression has occurred",
    },
    {
        "doc_id": "d04",
        "sentence": (
            "If tariffs on Asian-fabricated controllers were imposed at the rates now under "
            "discussion, our hardware cost of goods could rise by three to four points."
        ),
        "why": "counterfactual conditional - tariffs have not been imposed",
    },
    {
        "doc_id": "d05",
        "sentence": (
            "The company said it has no plans to divest the Dresden facility, "
            "notwithstanding recent press reports to the contrary."
        ),
        "why": "negated intent, reported speech - denies a divestiture rumor",
    },
    {
        "doc_id": "d06",
        "sentence": (
            "Duarte has acknowledged that the chipmaker may face fines if the Commission "
            "ultimately finds against it, though no finding has been made and no statement "
            "of objections has been issued."
        ),
        "why": "hypothetical plus explicit negation - no finding, no fine exists",
    },
    {
        "doc_id": "d07",
        "sentence": (
            "The Company does not expect the SEC inquiry to delay that review, although it "
            "may do so if additional document requests are issued."
        ),
        "why": "negated expectation followed by a hedged conditional reversal",
    },
    {
        "doc_id": "d09",
        "sentence": (
            "One noted that if diesel remains above $4.40 a gallon into the fourth quarter, "
            "the revised margin floor of the Company could prove optimistic."
        ),
        "why": "hypothetical, attributed to an unnamed analyst - not a company statement",
    },
    {
        "doc_id": "d10",
        "sentence": (
            "We do not expect a decision this fiscal year, and if the Commission were to "
            "issue a statement of objections, we would respond within the standard deadline."
        ),
        "why": "negated expectation plus subjunctive conditional",
    },
]
