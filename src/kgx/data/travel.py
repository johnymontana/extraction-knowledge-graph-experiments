"""Synthetic travel corpus: bookings, itineraries, disruptions and expenses.

Everything in this module is FICTIONAL. Meridian Air (MR), Northwind Airways
(NW), the Marina Quay Hotel, the Kanda Rise Hotel, the Rheinblick Hof, the
Aldgate Rowe, The Longacre Midtown, Skyline Sedan Services, Kestrel Cars,
Typhoon Meilin, and every booking reference, ticket number, loyalty number and
fare in these twelve documents are invented. Priya Raman, Tomas Ferreira and
Dr. Elena Vasquez are invented people. Airports appear under their genuine IATA
codes (LHR, JFK, SIN, NRT, FRA, GRU) because a code is a fact about the world,
not a trademark, but no real carrier, hotel or airport operation is described.

The corpus is twelve documents dated 2026-03-02 through 2026-07-08, written to
be *hard* in one specific way that separates it from the other corpora here:

  * **Codes and controlled vocabularies.** Travel prose is half identifiers.
    "LHR" and "London Heathrow" refer to the same node and share no characters,
    so no amount of edit distance, embedding similarity or fuzzy blocking will
    merge them -- but one dictionary lookup in ``IATA_GAZETTEER`` resolves it
    exactly and for free. The same holds for "MR" / "Meridian Air", "SIN" /
    "Changi", "NRT" / "Narita". This is the corpus for demonstrating that the
    right resolver for a closed vocabulary is a *gazetteer*, not a model.
  * **Codes that look alike and are not the same thing.** ``MR117`` is a
    flight; ``MR7K2QX9`` is a booking reference; ``SKY-4471902`` is a loyalty
    number; ``MQH-77310`` is a hotel confirmation. All four are alphanumeric
    blobs beginning with letters, and a resolver that blocks on shape will
    happily merge them.
  * **Reused identifiers across time.** ``MR204`` operates Frankfurt to London
    in March (d02) and again in June (d11). Flight numbers are types, not
    instances; a store that treats "MR204" as one node has to be comfortable
    with that or model the leg date separately.
  * **Ambiguous short forms.** "Frankfurt" is used for the airport; "Singapore"
    is simultaneously a city, a country and (via SIN) an airport. "Northwind"
    is a deliberate cross-corpus hazard -- Northwind *Airways* here is not
    Northwind *Logistics* in ``kgx.data.documents``, though both are shortened
    to "Northwind" in running text.
  * **Cross-corpus link.** Priya Raman, Tomas Ferreira and Dr. Elena Vasquez
    are shared on purpose with ``kgx.data.conversations`` and
    ``kgx.data.documents``, so the travel graph can be joined to the memory and
    news graphs on the traveller. Note that ``conversations`` spells the name
    "Tomás Ferreira" with an accent and this corpus does not.

``GOLD_TRAVEL_FACTS`` is the hand-written answer key, expressed in the closed
relation vocabulary of the ``TRAVEL`` ontology in ``kgx.domains``.
``ALIAS_GROUPS`` is the gold clustering for entity resolution.
``IATA_GAZETTEER`` is the controlled vocabulary a dictionary resolver links
against, including four decoy codes that never occur in the text.
"""

DOCUMENTS: list[dict] = [
    {
        "doc_id": "t01",
        "date": "2026-03-02",
        "source": "Airline Booking Confirmation",
        "title": "Meridian Air — booking MR7K2QX9 confirmed",
        "text": (
            "Thank you for choosing Meridian Air. Your booking is confirmed.\n\n"
            "Booking reference: MR7K2QX9\n"
            "Passenger: Ms Priya Raman\n"
            "Meridian Skyline number: SKY-4471902 (Gold)\n\n"
            "Outbound — MR117, Thursday 12 March 2026. London Heathrow (LHR), Terminal 3, "
            "departing 21:45, arriving Singapore Changi (SIN), Terminal 1, at 17:55 the "
            "following day. Seat 14A.\n\n"
            "Return — MR118, Tuesday 17 March 2026. Singapore Changi (SIN) 23:20, arriving "
            "London Heathrow (LHR) 05:40 the next morning. Seat 12C.\n\n"
            "Both sectors are operated by Meridian Air; there is no codeshare on this "
            "itinerary. Meridian serves Changi twice daily from Heathrow, and the evening "
            "departure is the one your travel desk has standardised on.\n\n"
            "Priya Raman booked MR7K2QX9 on 2 March, and the reference covers MR117 and "
            "MR118. Ms Raman is travelling to attend the Asia-Pacific Logistics Summit, and "
            "the fare has been charged to the Platform Engineering travel budget. Baggage: two "
            "checked bags at 23 kg each, included under the Skyline Gold allowance.\n\n"
            "Online check-in for MR117 opens 24 hours before departure. Please carry a "
            "passport valid for at least six months beyond your date of entry to Singapore. "
            "A Singapore Arrival Card is required for entry to Singapore and must be "
            "submitted online within three days of arrival."
        ),
    },
    {
        "doc_id": "t02",
        "date": "2026-03-04",
        "source": "Itinerary Email",
        "title": "Trip file 2026-0311 — T. Ferreira, GRU–FRA–LHR",
        "text": (
            "Tomas — your travel file for the supplier audit is below. Please check the "
            "spelling of your name against your passport before Friday.\n\n"
            "Booking reference NW3F8LT2, issued on Northwind Airways stock, covers both "
            "flights.\n\n"
            "11 Mar  NW882  São Paulo Guarulhos (GRU) 22:10 → Frankfurt am Main (FRA) "
            "13:35 +1\n"
            "12 Mar  MR204  Frankfurt am Main (FRA) 16:20 → London Heathrow (LHR) 17:05\n\n"
            "The second leg is a Meridian Air aircraft ticketed on the Northwind stock, so "
            "the miles will earn on NW882 only. Mr Ferreira is a member of the Northwind "
            "Compass Club under number NWC-207714; if you would rather have the MR204 miles "
            "credited to Meridian Skyline, tell the desk before departure.\n\n"
            "You connect at Frankfurt am Main with five hours on the ground. The Rheinblick "
            "Hof is a ten-minute walk from the terminal and we have held a day room there "
            "under confirmation RBH-40921 in case the inbound runs late.\n\n"
            "In London you are booked at the Aldgate Rowe for three nights; it is a "
            "fifteen-minute walk from the office. A prepaid Heathrow Express transfer from "
            "LHR to Paddington is attached to the same trip file.\n\n"
            "Brazilian passport holders need an electronic travel authorisation for the "
            "United Kingdom; yours runs to November, so no action is needed. T. Ferreira is "
            "back in São Paulo on the 16th."
        ),
    },
    {
        "doc_id": "t03",
        "date": "2026-03-11",
        "source": "Hotel Reservation",
        "title": "Marina Quay Hotel — reservation MQH-77310 for P. Raman",
        "text": (
            "MARINA QUAY HOTEL, SINGAPORE\n"
            "Reservation confirmation MQH-77310\n\n"
            "Guest: P. Raman\n"
            "Arrival: Friday 13 March 2026, from 14:00\n"
            "Departure: Tuesday 17 March 2026, by 11:00\n"
            "Room: one King Harbour View, four nights, non-smoking, high floor requested\n"
            "Rate: SGD 412 per night, room only, prepaid and non-refundable\n\n"
            "The Marina Quay Hotel is on Marina Boulevard in Singapore, eleven minutes by "
            "car from Singapore Changi and directly above the Bayfront interchange. Guests "
            "on late inbound flights may check in at any hour; the front desk is staffed "
            "overnight.\n\n"
            "We have noted that you arrive on MR117 on the 13th and that your reason for "
            "travel is the Asia-Pacific Logistics Summit, which is being held at the "
            "convention centre next door. The summit delegate rate has been applied.\n\n"
            "A prepaid transfer with Skyline Sedan Services has been arranged from Changi "
            "to the hotel and is included in this reservation. Skyline Sedan Services "
            "operates across Singapore; the driver will meet you in the Terminal 1 arrivals "
            "hall. If your flight is delayed the transfer tracks SIN arrivals and re-times "
            "itself.\n\n"
            "Please present the passport used at booking on arrival. The hotel is required "
            "to record the document number of every foreign guest."
        ),
    },
    {
        "doc_id": "t04",
        "date": "2026-03-12",
        "source": "Disruption Notice",
        "title": "Meridian Air: MR117 cancelled — you have been rebooked",
        "text": (
            "Ms Raman, we are sorry. MR117 to Singapore this evening has been cancelled by "
            "an emergency runway resurfacing closure at London Heathrow.\n\n"
            "The runway resurfacing closure has taken roughly a third of tonight's long-haul "
            "departures out of the schedule, and booking MR7K2QX9 is disrupted by it in "
            "full. Heathrow expects to release the northern runway at 06:00 tomorrow. This "
            "is not a weather event, and no Meridian Air departure after 13 March is "
            "affected.\n\n"
            "We have protected you on Northwind Airways:\n\n"
            "  NW441  LHR 07:15 → FRA 09:55, Friday 13 March\n"
            "  NW443  FRA 12:40 → SIN 06:20, Saturday 14 March\n\n"
            "New booking reference NW6R1XKD. NW441 takes the place of MR117 on the "
            "outbound, and the reissue replaces MR7K2QX9 in full; the original ticket has "
            "been voided and the Marina Quay Hotel has been told you now arrive a day "
            "late. You connect at Frankfurt am Main with two hours and forty-five minutes "
            "on the ground.\n\n"
            "Seats are assigned, but Northwind cannot honour your Skyline Gold seating "
            "preferences, and NW is a partner rather than a Skyline carrier, so miles post "
            "at 50%.\n\n"
            "If you would rather wait and take MR117 on the 14th, reply CANCEL REBOOK by "
            "23:00 and we will reinstate the original itinerary. Compensation for the "
            "cancellation of Meridian Air 117 has been raised automatically against "
            "MR7K2QX9."
        ),
    },
    {
        "doc_id": "t05",
        "date": "2026-04-02",
        "source": "Loyalty Statement",
        "title": "Meridian Skyline — Q1 2026 statement for member SKY-4471902",
        "text": (
            "MERIDIAN SKYLINE — QUARTERLY STATEMENT\n"
            "Member: Priya Raman\n"
            "Membership number: SKY-4471902\n"
            "Tier: Gold, held through 31 March 2027\n\n"
            "Meridian Skyline is the frequent flyer programme operated by Meridian Air. "
            "This statement covers 1 January to 31 March 2026.\n\n"
            "  Opening balance                          61,340\n"
            "  Flown miles, MR118 SIN–LHR               +6,760\n"
            "  Partner credit, NW441 and NW443          +3,210\n"
            "  Goodwill award, MR 117 cancellation      +5,000\n"
            "  Redeemed, seat upgrade on MR118          -9,500\n"
            "  Closing balance                          66,810\n\n"
            "You are 4,190 tier miles short of Platinum. Two more Meridian long-haul "
            "sectors before 30 September would carry you over.\n\n"
            "Northwind Airways credits at partner rates only. The Northwind Compass Club "
            "account held by Tomas Ferreira is a separate programme and cannot be pooled "
            "with Skyline.\n\n"
            "Elena Vasquez has been added as a nominated redemption beneficiary on this "
            "account at your request. Dr Vasquez may book award tickets against your "
            "balance up to 40,000 miles per year.\n\n"
            "Skyline Gold entitles you to lounge access at Heathrow, Changi and Frankfurt "
            "am Main, two extra checked bags, and priority standby on any MR flight."
        ),
    },
    {
        "doc_id": "t06",
        "date": "2026-04-09",
        "source": "Travel Document Reminder",
        "title": "Action required before Tokyo — documents for Dr E. Vasquez",
        "text": (
            "Dr Vasquez, three document items need clearing before the Tokyo research "
            "collaboration in May.\n\n"
            "1. Passport. Yours expires on 2 October 2026. Japan requires a passport valid "
            "for the duration of stay, so it will technically be accepted, but Northwind "
            "Airways has refused boarding on thinner margins than this and we would rather "
            "you renewed. Allow four weeks.\n\n"
            "2. Visa. A single-entry short-stay visa is required for entry to Japan on the "
            "academic sponsorship you are travelling under, and the sponsor letter from the "
            "Kanda institute is the supporting document. This is not the same as the waiver "
            "you used for the Singapore leg last year.\n\n"
            "3. Onward evidence. Immigration at Tokyo Narita has been asking for proof of "
            "onward travel. Booking NWQ9ZP41 already contains the return sector, so print "
            "the itinerary and carry it.\n\n"
            "Nothing is required on the United Kingdom side. Elena Vasquez holds a British "
            "passport and exit checks at Heathrow are automatic.\n\n"
            "Your outbound is NW612 from London Heathrow to Tokyo Narita on 6 May, operated "
            "by Northwind Airways throughout with no Meridian codeshare. Northwind now "
            "serves Narita daily rather than five times a week, so a same-day rebooking is "
            "realistic if anything slips."
        ),
    },
    {
        "doc_id": "t07",
        "date": "2026-04-21",
        "source": "Expense Report",
        "title": "Expense claim ER-2026-0418 — P. Raman, Singapore, 12–18 March",
        "text": (
            "EXPENSE CLAIM ER-2026-0418\n"
            "Claimant: P. Raman, Platform Engineering\n"
            "Trip: Asia-Pacific Logistics Summit, Singapore, 12–18 March 2026\n\n"
            "  Airfare, Meridian Air, booking MR7K2QX9          GBP 1,842.00  refunded\n"
            "  Airfare, Northwind Airways, booking NW6R1XKD     GBP 2,117.40  approved\n"
            "  Hotel, Marina Quay Hotel, MQH-77310, 4 nights    SGD 1,648.00\n"
            "  Transfer, Skyline Sedan Services, Changi–hotel   SGD    78.00\n"
            "  Rail, Heathrow Express, Paddington–LHR           GBP    25.00\n"
            "  Meals and incidentals, six days                  GBP   312.55\n"
            "  Summit delegate fee                              GBP   695.00\n\n"
            "Notes for finance. The Meridian fare booked under MR7K2QX9 was refunded in "
            "full after the cancellation, so only the Northwind reissue under NW6R1XKD is "
            "claimable; both lines are shown so the audit trail matches the ticket "
            "numbers.\n\n"
            "The hotel invoice includes one unused night. The rebooked routing through "
            "Frankfurt am Main arrived on 14 March rather than 13 March and the Marina Quay "
            "rate was non-refundable. Ms Raman travelled in economy on both carriers; no "
            "upgrade was purchased or redeemed.\n\n"
            "The delegate fee is a summit cost rather than a travel cost and has been coded "
            "to the conference budget. Approver: T. Ferreira."
        ),
    },
    {
        "doc_id": "t08",
        "date": "2026-04-28",
        "source": "Airline Booking Confirmation",
        "title": "Northwind Airways e-ticket NWQ9ZP41 — VASQUEZ/ELENA DR",
        "text": (
            "NORTHWIND AIRWAYS — E-TICKET RECEIPT\n"
            "Record locator: NWQ9ZP41\n"
            "Passenger: VASQUEZ/ELENA DR\n\n"
            "NW612  06 May 2026  London Heathrow (LHR) T2 11:40 → Tokyo Narita (NRT) T1 "
            "07:35 +1\n"
            "NW613  15 May 2026  Tokyo Narita (NRT) T1 09:50 → London Heathrow (LHR) T2 "
            "14:25\n\n"
            "Both sectors are operated by Northwind Airways. Record locator NWQ9ZP41 covers "
            "NW612, NW613 and a Narita Express ticket. Tokyo Narita is roughly sixty "
            "kilometres east of Tokyo, and the Narita Express runs from the airport to "
            "Tokyo Station; collect the ticket at the JR counter after immigration.\n\n"
            "Dr. Elena Vasquez is booked at the Kanda Rise Hotel in Tokyo, Japan, under "
            "hotel confirmation KRH-88214, nine nights, arriving 7 May. The hotel is a "
            "four-minute walk from Kanda station.\n\n"
            "Purpose of travel is recorded as the Tokyo research collaboration. The trip is "
            "funded centrally rather than from a departmental budget, and the fare rules "
            "permit one free change per sector up to three hours before departure.\n\n"
            "Japan requires a visa for this itinerary; do not travel without it. Dr. Elena "
            "Vasquez holds a British passport and a single-entry short-stay visa, and "
            "Northwind will deny boarding at London Heathrow if the visa page cannot be "
            "shown at the gate. Miles for both sectors will be credited to Meridian Skyline "
            "as a partner award."
        ),
    },
    {
        "doc_id": "t09",
        "date": "2026-05-07",
        "source": "Disruption Notice",
        "title": "Typhoon Meilin — NRT ground stop, Narita Express suspended",
        "text": (
            "Operations bulletin, 07:10 local.\n\n"
            "Typhoon Meilin made landfall south of Chiba overnight. Tokyo Narita closed "
            "both runways at 04:00 and is running a ground stop until at least 14:00.\n\n"
            "NW612 from London Heathrow was delayed by Typhoon Meilin and was airborne when "
            "the stop was called. It held for ninety minutes, landed in the first window, "
            "and is now on stand at Tokyo Narita; disembarkation has begun. The delay to "
            "NW612 is running at two hours fifty.\n\n"
            "The Narita Express has been suspended by Typhoon Meilin in both directions and "
            "JR East has published no restart time. Northwind Airways is running "
            "replacement coaches from "
            "Terminal 1 to Tokyo Station for ticketed passengers, journey time three hours "
            "rather than the usual one.\n\n"
            "Dr Vasquez is on this flight. Vasquez has been advised to take the coach "
            "rather than wait for the train, and the Kanda Rise Hotel has agreed to hold "
            "KRH-88214 for a late arrival. Nothing in booking NWQ9ZP41 needs reissuing; the "
            "return sector NW613 on 15 May is unaffected.\n\n"
            "Two Meridian Air services into Tokyo Narita were cancelled outright, and "
            "MR455, tomorrow's Heathrow departure, has been retimed by four hours. Meridian "
            "will publish a revised schedule at 18:00."
        ),
    },
    {
        "doc_id": "t10",
        "date": "2026-05-19",
        "source": "Itinerary Email",
        "title": "New York, 2–6 June — booking MR2H8VD6",
        "text": (
            "Priya — the New York trip is ticketed. Booking reference MR2H8VD6, Meridian "
            "Air.\n\n"
            "MR310  Tue 02 Jun  London Heathrow (LHR) T3 10:05 → John F. Kennedy (JFK) T4 "
            "13:15\n"
            "MR311  Sat 06 Jun  John F. Kennedy (JFK) T4 18:40 → London Heathrow (LHR) T3 "
            "06:30 +1\n\n"
            "Nonstop both ways, no connection, both operated by Meridian Air. MR has just "
            "added a second daily rotation to John F. Kennedy, which is why the morning "
            "departure exists at all — Meridian only started serving New York in "
            "November.\n\n"
            "Hotel: The Longacre Midtown in New York, five nights, confirmation LMT-51204. "
            "Ms Raman has stayed there twice before and the property holds her preferences "
            "on file.\n\n"
            "Ground: a Kestrel Cars sedan will meet the arrival at JFK. Kestrel Cars "
            "operates throughout New York, in the United States, and the driver's number "
            "will be texted to you an hour before landing.\n\n"
            "Purpose is the Halcyon customer onsite rather than the summit series, so this "
            "one is coded to the account team's budget.\n\n"
            "Two document notes. Your electronic travel authorisation for the United States "
            "was renewed in February and runs to 2028. Your passport expires in January "
            "2027, and the six-month rule bites from July, so renew it before the next "
            "Singapore trip."
        ),
    },
    {
        "doc_id": "t11",
        "date": "2026-06-11",
        "source": "Disruption Notice",
        "title": "Frankfurt ATC walkout — MR204 delayed, connection lost",
        "text": (
            "Mr Ferreira — Frankfurt is a mess today and your connection is gone.\n\n"
            "German air traffic control staff began a twenty-four hour walkout at 05:00. "
            "Frankfurt am Main, in Germany, is releasing roughly forty per cent of its "
            "scheduled movements. NW882 from São Paulo Guarulhos landed on time at 13:35, "
            "but MR204 to London Heathrow has been delayed to 22:50 by the walkout. Booking "
            "NW3F8LT2 is disrupted by the walkout and MR204 will not be called before "
            "then.\n\n"
            "We have moved you to MR206, which departs Frankfurt am Main at 07:40 tomorrow "
            "and reaches London Heathrow at 08:25. MR206 replaces MR204 on booking "
            "NW3F8LT2. The coupon has been reissued rather than voided, so your Northwind "
            "Compass Club credit for NW882 is unaffected.\n\n"
            "A room is held for you tonight at the Rheinblick Hof, the same property as the "
            "March trip, under confirmation RBH-61088. Tonight's per diem is the eurozone "
            "rate rather than the Brazilian one.\n\n"
            "Tomas, if you would rather sit it out and take MR204 tonight, call the desk "
            "before 20:00. Be aware that the Aldgate Rowe in London holds your room from "
            "tonight either way and the first night is non-refundable."
        ),
    },
    {
        "doc_id": "t12",
        "date": "2026-07-08",
        "source": "Trip Summary",
        "title": "H1 2026 travel summary — Platform Engineering",
        "text": (
            "Half-year travel summary, Platform Engineering, January to June 2026.\n\n"
            "Three people travelled. Priya Raman took two long-haul trips: Singapore in "
            "March for the Asia-Pacific Logistics Summit, and New York in June for the "
            "Halcyon customer onsite. Tomas Ferreira flew Sao Paulo to London twice for the "
            "supplier audit, routing through Frankfurt am Main both times. Dr. Elena "
            "Vasquez spent nine days in Tokyo on the research collaboration.\n\n"
            "Destinations this half were London, Singapore, Tokyo and New York. London is "
            "in the United Kingdom, Tokyo is in Japan, and New York is in the United "
            "States; all four are on the standard per-diem schedule.\n\n"
            "Carrier split by sector: Meridian Air eleven, Northwind Airways six. Meridian "
            "remains the preferred carrier out of London Heathrow and holds the Skyline "
            "relationship. Northwind is used where MR has no route, principally to "
            "Narita.\n\n"
            "Two disruptions cost money. The runway closure at Heathrow on 12 March "
            "cancelled MR117 and forced the reissue from MR7K2QX9 to NW6R1XKD, a net GBP "
            "275 plus one unused night at the Marina Quay Hotel. The Frankfurt walkout in "
            "June cost a night at the Rheinblick Hof and a day of Ferreira's time.\n\n"
            "Document risk: Raman's passport expires in January 2027 and Vasquez holds one "
            "expiring in October 2026. Renewals are open on both."
        ),
    },
]


# (head_canonical, relation, tail_canonical) triples a perfect pipeline should
# recover from DOCUMENTS. The relation vocabulary is closed -- see TRAVEL in
# kgx.domains -- and all sixteen relations are represented here.
#
# This key is a REPRESENTATIVE SAMPLE, not an exhaustive enumeration: the corpus
# supports roughly half again as many true triples (every route endpoint, every
# located_in edge the gazetteer knows). Score recall against it directly; for
# precision, treat an extracted triple whose entities are in the corpus but
# whose edge is absent here as unjudged rather than wrong.
GOLD_TRAVEL_FACTS: list[tuple[str, str, str]] = [
    # --- who holds which reservation ---------------------------------------
    ("Priya Raman", "booked", "MR7K2QX9"),
    ("Tomas Ferreira", "booked", "NW3F8LT2"),
    ("Dr. Elena Vasquez", "booked", "NWQ9ZP41"),
    # --- what a reservation contains ---------------------------------------
    ("MR7K2QX9", "covers", "MR117"),
    ("NWQ9ZP41", "covers", "Narita Express"),
    # --- who is on which aircraft ------------------------------------------
    ("Priya Raman", "flies_on", "MR117"),
    ("Dr. Elena Vasquez", "flies_on", "NW612"),
    # --- carrier of a flight or a programme ---------------------------------
    ("MR117", "operated_by", "Meridian Air"),
    ("Meridian Skyline", "operated_by", "Meridian Air"),
    # --- route endpoints ----------------------------------------------------
    ("MR117", "departs_from", "London Heathrow"),
    ("NW882", "departs_from", "São Paulo Guarulhos"),
    ("MR117", "arrives_at", "Singapore Changi"),
    ("NW612", "arrives_at", "Tokyo Narita"),
    # --- transfer points ----------------------------------------------------
    ("NW6R1XKD", "connects_through", "Frankfurt am Main"),
    # --- accommodation ------------------------------------------------------
    ("Priya Raman", "stays_at", "Marina Quay Hotel"),
    ("Dr. Elena Vasquez", "stays_at", "Kanda Rise Hotel"),
    # --- geography (the gazetteer's job, stated in text at least once) ------
    ("London Heathrow", "located_in", "London"),
    ("Kanda Rise Hotel", "located_in", "Tokyo"),
    ("New York", "located_in", "United States"),
    # --- loyalty ------------------------------------------------------------
    ("Priya Raman", "member_of", "Meridian Skyline"),
    ("Tomas Ferreira", "member_of", "Northwind Compass Club"),
    # --- route network, not a single flight ---------------------------------
    ("Northwind Airways", "serves", "Tokyo Narita"),
    # --- why the trip happened ----------------------------------------------
    ("Priya Raman", "travels_for", "Asia-Pacific Logistics Summit"),
    ("Dr. Elena Vasquez", "travels_for", "Tokyo research collaboration"),
    # --- documents ----------------------------------------------------------
    ("Dr. Elena Vasquez", "holds_document", "single-entry short-stay visa"),
    ("Priya Raman", "holds_document", "passport"),
    ("Singapore Arrival Card", "required_for", "Singapore"),
    # --- things going wrong -------------------------------------------------
    ("MR117", "disrupted_by", "runway resurfacing closure"),
    ("NW612", "disrupted_by", "Typhoon Meilin"),
    ("Narita Express", "disrupted_by", "Typhoon Meilin"),
    # --- rebooking supersedes the original ----------------------------------
    ("NW6R1XKD", "replaces", "MR7K2QX9"),
    ("NW441", "replaces", "MR117"),
    ("MR206", "replaces", "MR204"),
]


# Gold clustering for entity resolution. canonical -> every surface form that
# literally occurs in some DOCUMENTS[i]["text"]. The airport and airline groups
# are the interesting ones: "LHR" and "London Heathrow" have zero characters in
# common, so string similarity cannot merge them and IATA_GAZETTEER must.
ALIAS_GROUPS: dict[str, list[str]] = {
    # -- airports (code vs. name vs. short name) ---------------------------
    "London Heathrow": [
        "London Heathrow",
        "Heathrow",
        "LHR",
    ],
    "Singapore Changi": [
        "Singapore Changi",
        "Changi",
        "SIN",
    ],
    "Tokyo Narita": [
        "Tokyo Narita",
        "Narita",
        "NRT",
    ],
    "Frankfurt am Main": [
        "Frankfurt am Main",
        "Frankfurt",
        "FRA",
    ],
    "John F. Kennedy": [
        "John F. Kennedy",
        "JFK",
    ],
    "São Paulo Guarulhos": [
        "São Paulo Guarulhos",
        "GRU",
    ],
    # -- airlines ----------------------------------------------------------
    "Meridian Air": [
        "Meridian Air",
        "Meridian",
        "MR",
    ],
    "Northwind Airways": [
        "Northwind Airways",
        "Northwind",
        "NW",
    ],
    # -- travellers --------------------------------------------------------
    "Priya Raman": [
        "Priya Raman",
        "P. Raman",
        "Ms Raman",
        "Priya",
        "Raman",
    ],
    "Tomas Ferreira": [
        "Tomas Ferreira",
        "T. Ferreira",
        "Mr Ferreira",
        "Tomas",
        "Ferreira",
    ],
    "Dr. Elena Vasquez": [
        "Dr. Elena Vasquez",
        "Elena Vasquez",
        "Dr Vasquez",
        "Vasquez",
    ],
    # -- flights (a flight number is a type, and MR204 recurs in June) -----
    "MR117": [
        "MR117",
        "MR 117",
        "Meridian Air 117",
    ],
    # -- loyalty programmes and their numbers ------------------------------
    "Meridian Skyline": [
        "Meridian Skyline",
        "Skyline",
        "SKY-4471902",
    ],
    "Northwind Compass Club": [
        "Northwind Compass Club",
        "Compass Club",
        "NWC-207714",
    ],
    # -- places ------------------------------------------------------------
    "São Paulo": [
        "São Paulo",
        "Sao Paulo",
    ],
    # -- hotels and their confirmation codes -------------------------------
    "Marina Quay Hotel": [
        "Marina Quay Hotel",
        "Marina Quay",
        "MQH-77310",
    ],
    "Kanda Rise Hotel": [
        "Kanda Rise Hotel",
        "KRH-88214",
    ],
    "Rheinblick Hof": [
        "Rheinblick Hof",
        "RBH-40921",
        "RBH-61088",
    ],
    "The Longacre Midtown": [
        "The Longacre Midtown",
        "LMT-51204",
    ],
}


# The controlled vocabulary a dictionary/gazetteer resolver links against.
# Keyed by the code exactly as it appears in text, because the code is the join
# key that fuzzy matching cannot recover. ``aliases`` lists the surface forms a
# lookup table should index; ``kind`` separates the two namespaces, which
# matters because two-letter airline codes and three-letter airport codes are
# drawn from overlapping alphabets and MR is both a carrier and the prefix of
# every Meridian flight number and booking reference.
#
# DXB, ICN, CDG and ZQ are decoys: they are valid vocabulary entries that never
# occur anywhere in DOCUMENTS. A resolver that "finds" them is hallucinating,
# and a gazetteer benchmark without negatives cannot show that.
IATA_GAZETTEER: dict[str, dict] = {
    # -- airports that occur in the corpus ---------------------------------
    "LHR": {
        "name": "London Heathrow",
        "city": "London",
        "country": "United Kingdom",
        "kind": "airport",
        "aliases": ["London Heathrow", "Heathrow"],
    },
    "JFK": {
        "name": "John F. Kennedy",
        "city": "New York",
        "country": "United States",
        "kind": "airport",
        "aliases": ["John F. Kennedy", "JFK International", "Kennedy"],
    },
    "SIN": {
        "name": "Singapore Changi",
        "city": "Singapore",
        "country": "Singapore",
        "kind": "airport",
        "aliases": ["Singapore Changi", "Changi"],
    },
    "NRT": {
        "name": "Tokyo Narita",
        "city": "Tokyo",
        "country": "Japan",
        "kind": "airport",
        "aliases": ["Tokyo Narita", "Narita"],
    },
    "FRA": {
        "name": "Frankfurt am Main",
        "city": "Frankfurt",
        "country": "Germany",
        "kind": "airport",
        "aliases": ["Frankfurt am Main", "Frankfurt"],
    },
    "GRU": {
        "name": "São Paulo Guarulhos",
        "city": "São Paulo",
        "country": "Brazil",
        "kind": "airport",
        "aliases": ["São Paulo Guarulhos", "Sao Paulo Guarulhos", "Guarulhos"],
    },
    # -- airlines that occur in the corpus ---------------------------------
    "MR": {
        "name": "Meridian Air",
        "country": "United Kingdom",
        "kind": "airline",
        "hub": "LHR",
        "loyalty_program": "Meridian Skyline",
        "aliases": ["Meridian Air", "Meridian"],
    },
    "NW": {
        "name": "Northwind Airways",
        "country": "United States",
        "kind": "airline",
        "hub": "JFK",
        "loyalty_program": "Northwind Compass Club",
        "aliases": ["Northwind Airways", "Northwind"],
    },
    # -- decoys: valid entries that never appear in DOCUMENTS ---------------
    "DXB": {
        "name": "Dubai International",
        "city": "Dubai",
        "country": "United Arab Emirates",
        "kind": "airport",
        "aliases": ["Dubai International", "Dubai"],
    },
    "ICN": {
        "name": "Seoul Incheon",
        "city": "Seoul",
        "country": "South Korea",
        "kind": "airport",
        "aliases": ["Seoul Incheon", "Incheon"],
    },
    "CDG": {
        "name": "Charles de Gaulle",
        "city": "Paris",
        "country": "France",
        "kind": "airport",
        "aliases": ["Charles de Gaulle", "Roissy"],
    },
    "ZQ": {
        "name": "Zephyr Continental",
        "country": "Canada",
        "kind": "airline",
        "hub": "DXB",
        "loyalty_program": "Zephyr Horizon",
        "aliases": ["Zephyr Continental", "Zephyr"],
    },
}
