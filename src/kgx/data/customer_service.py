"""Synthetic customer-support corpus for the knowledge-graph extraction demo.

Everything in this module is FICTIONAL. The Aurora 14, the Halcyon Buds and the
Northwind Router N600 are invented products; Kestrel Compute, Solent Audio,
Blackmere Direct and Cobalt Networks are invented competitors; Priya Raman,
Marcus Webb, Ana Duarte, Dev Shah and Tomas Ferreira are invented people. The
order numbers, refund references, prices and warranty terms are all made up. No
real company's support process is described.

The corpus is eight support threads dated 2026-03-02 through 2026-06-08, across
three channels (email, chat, phone log). It exists to exercise two things that
the news corpus in :mod:`kgx.data.documents` does not:

  * **Conversational structure.** A support thread is not a document with a
    narrator. The subject of most facts is a pronoun or a definite description
    whose antecedent is in an *earlier turn, usually spoken by the other party*:
    "It clicks. The LED does one long blink instead of two." Feeding the whole
    thread to the extractor and hoping is the failure mode this corpus is built
    to expose -- ``PRONOUN_TARGETS`` lists the specific turns where a fact is
    unrecoverable without coreference. Compare the pipeline with and without
    :class:`kgx.coref.ConversationPreprocessor`; the gap is the point.
  * **Per-thread classification.** Intent, priority and resolution state are
    properties of the *thread*, not spans in it. No amount of entity extraction
    yields them: nobody writes "priority: urgent". They need a separate
    classification pass, scored against ``GOLD_INTENTS``.

Written to be hard in the usual checkable ways:

  * Surface-form variation. "Aurora 14" / "Aurora-14s" / "the Aurora" / "the
    laptop"; "Northwind Router N600" / "N600" / "the router"; "Halcyon Buds" /
    "the Buds" / "the earphones"; order "A-99213" / "#A-99213" / "order A99213";
    "B-40877" / "#B-40877" / "order B40877". Agents introduce themselves in full
    once and are first-name-only afterwards. ``ALIAS_GROUPS`` is the gold
    clustering, and every string in it occurs literally in some turn.
  * Resolution supersession. Three threads change their remedy mid-conversation:
    t01 offers an advance replacement and ends in a refund, t02 pushes a
    firmware update and ends in a replacement, t05 books a replacement router
    and cancels it for a refund. A pipeline that stores every remedy it sees as
    a live fact ends up asserting that Priya both received a replacement laptop
    and was refunded for it -- and it does: over t01, ``resolved_by`` returns
    BOTH ('shutdown fault' -> 'refund', 0.69) and ('shutdown fault' -> 'advance
    replacement', 0.53) from the same thread. Only the ``replaces`` edge tells
    you which one survives. The ``replaces`` triples in ``GOLD_CS_FACTS`` are
    the supersession key; the same invalidation machinery as
    ``kgx.data.conversations.CONTRADICTIONS`` applies.
  * Cross-corpus hazard, deliberately. "Northwind" here is a *router brand* and
    "Halcyon" is an *earphone brand*, while in :mod:`kgx.data.documents` they
    are a freight company and a chipmaker. Priya Raman, Marcus Webb and Ana
    Duarte are executives there and customers here; Dev Shah and "Tomás
    Ferreira" are the user's colleagues in :mod:`kgx.data.conversations` and
    support agents here -- note the accent, dropped in this corpus. Merging
    across corpora on name alone is exactly the over-merge the resolution
    notebook should surface rather than hide.

Measured baseline, gliner2.5-base-v1 against ``kgx.domains.CUSTOMER_SERVICE``,
speaker turns rendered as "Priya Raman: ..." / "Dev Shah: ...":

  * 4-turn windows (stride 2, 46 windows): all 13 entity types and 15 of 16
    relations fire, 0 edges rejected by ``DocGraph.validate``.
  * Single turns: ``replaces`` fires -- correctly, ('full refund' -> 'overnight
    replacement', 0.49) in t05 -- and ``about_order``, ``breaches`` and
    ``resolved_by`` stop. The extraction unit is per relation, not global; union
    the two granularities.
  * Do NOT feed a whole thread in one call. t01 is 517 words and trips the
    ``kgx.extract`` long-input warning; relation recall over the full thread is
    visibly worse than over its windows.

Three things about the writing are deliberate affordances rather than accidents,
and the notebook should say so:

  * The agent restates the customer's clause as a noun phrase ("Case CS-10231 --
    unexpected shutdown fault, reported by Priya Raman"). Without that
    nominalisation, every relation headed by ``issue`` measures zero, because
    GLiNER's span head wants a noun phrase and a fault is stated as a clause.
    Threads t04, t06 and t07 have no such restatement, on purpose.
  * Customers name rivals with an explicit switching cue ("your competitor
    Kestrel Compute", "we have already switched the office to Cobalt Networks").
    Without one, the rival's name types as ``product``, not ``competitor`` --
    measured on the earlier draft of these same turns.
  * The supersession in t02 and t05 is its own short agent message. That is how
    people actually use chat, and it is also the only granularity at which
    ``replaces`` survives.

``GOLD_CS_FACTS`` is the hand-written answer key, expressed with canonical
entity names rather than surface forms. It is a curated key, not an exhaustive
enumeration: every relation appears at least twice, and it deliberately includes
facts the measured baseline misses (all three ``replaces`` edges, most of
``about_order``).
"""

THREADS: list[dict] = [
    {
        "thread_id": "t01",
        "opened": "2026-03-02",
        "channel": "email",
        "subject": "Aurora 14 shutting down at 40% battery",
        "turns": [
            {
                "speaker": "customer",
                "text": (
                    "I bought an Aurora 14 in April last year — order A-99213 — and for "
                    "about three weeks now it has been shutting off without warning at "
                    "around 40% charge. No blue screen, no warning, the display just goes "
                    "black. Nothing has changed at my end: same charger, same desk, same "
                    "two applications open.\n\n"
                    "Priya Raman\nAccount 44120, Priority Care"
                ),
            },
            {
                "speaker": "agent",
                "text": (
                    "Hi Priya, this is Dev Shah in hardware support. Thank you for the "
                    "detail — that is more than most people give me. A hard cut at a "
                    "consistent percentage usually points at the battery rather than at the "
                    "operating system. Could you run the built-in power report and send me "
                    "design capacity against full charge capacity?\n\n"
                    "Case CS-10231 — unexpected shutdown fault, reported by Priya Raman, "
                    "Priority Care."
                ),
            },
            {
                "speaker": "customer",
                "text": "Design capacity 71.2 Wh. Full charge capacity 39.8 Wh. Cycle count 214.",
            },
            {
                "speaker": "agent",
                "text": (
                    "That is 56% of design at 214 cycles, well outside what we would expect. "
                    "The battery is the fault here: the pack collapses under load and the "
                    "firmware cuts power rather than brown out. Order A-99213 is still inside "
                    "the 12-month limited warranty, and the battery is covered as a service "
                    "part."
                ),
            },
            {
                "speaker": "customer",
                "text": (
                    "Good. How quickly can somebody actually look at it? I am on Priority "
                    "Care, and the two-business-day depot turnaround is the only reason I "
                    "pay for it."
                ),
            },
            {
                "speaker": "agent",
                "text": (
                    "Understood. I have raised a depot repair against order A-99213 and a "
                    "prepaid label goes out to you this afternoon. Priority Care puts the "
                    "machine at the front of the bench queue when it lands."
                ),
            },
            {
                "speaker": "customer",
                "text": (
                    "It is nine business days later and nothing has happened. I posted it the "
                    "morning after we spoke and the tracking says it was signed for on the "
                    "fifth. No call, no email, and the portal still says in transit."
                ),
            },
            {
                "speaker": "agent",
                "text": (
                    "I am sorry. I can see it sitting at the Reading depot, received and "
                    "never booked in. That is a clear miss against the two-business-day "
                    "turnaround and I am not going to defend it. Rather than leave you "
                    "waiting on the bench, I would like to send you an advance replacement "
                    "unit tomorrow and write off the one you posted."
                ),
            },
            {
                "speaker": "customer",
                "text": (
                    "Honestly, I do not want another one. Your competitor Kestrel Compute "
                    "build a machine with the same ports and a battery I would actually "
                    "trust, and the next six machines for my team were going to be "
                    "Aurora-14s. Refund the laptop and I will take my chances."
                ),
            },
            {
                "speaker": "agent",
                "text": (
                    "That is fair. We offered an advance replacement first. That is now "
                    "cancelled; a full refund takes its place. £1,249 back to the card you "
                    "paid on, five to seven working days, on order A99213. That resolves the "
                    "shutdown fault and closes the warranty claim."
                ),
            },
            {
                "speaker": "customer",
                "text": (
                    "Do it. For the record, this is about the depot and not about you. You "
                    "were the only part of this that worked."
                ),
            },
            {
                "speaker": "agent",
                "text": (
                    "Appreciated. The refund is authorised under reference RF-2261 and I have "
                    "flagged the Reading backlog to my supervisor. If the credit has not "
                    "landed by the twenty-fourth, reply to this email and it comes straight "
                    "back to me."
                ),
            },
        ],
    },
    {
        "thread_id": "t02",
        "opened": "2026-03-11",
        "channel": "chat",
        "subject": "left Halcyon Bud keeps dropping out",
        "turns": [
            {
                "speaker": "customer",
                "text": (
                    "hi — I tried the phone line twice this morning and gave up, so chat it "
                    "is. bought a pair of Halcyon Buds six weeks ago, order #B-40877. the "
                    "left one keeps dropping out. the right one is perfect."
                ),
            },
            {
                "speaker": "agent",
                "text": (
                    "Hi Marcus, Tomas Ferreira here. Sorry about the wait on the phones. "
                    "Logging it now as a dropout fault, reported by Marcus Webb, ref "
                    "CS-10288. When the left one drops, does it come back on its own or do "
                    "you have to put it in the case?"
                ),
            },
            {
                "speaker": "customer",
                "text": (
                    "it comes back after about ten seconds on its own. happens every two or "
                    "three minutes when I'm walking and almost never when I'm sitting at a "
                    "desk."
                ),
            },
            {
                "speaker": "agent",
                "text": (
                    "That pattern is the antenna handoff rather than a dead radio — the left "
                    "earbud is losing the primary role mid-stride. What firmware are the Buds "
                    "on? It is under About in the companion app."
                ),
            },
            {"speaker": "customer", "text": "2.1.4"},
            {
                "speaker": "agent",
                "text": (
                    "That explains it. Firmware 2.1.6 changed exactly this behaviour. Put both "
                    "in the charging case, leave the lid open, and keep the app in the "
                    "foreground for ten minutes; it will pull 2.1.6 down in the background."
                ),
            },
            {
                "speaker": "customer",
                "text": (
                    "did that yesterday. it's on 2.1.6 now and it still cuts out — four drops "
                    "on a twenty minute walk, which is worse than it was before."
                ),
            },
            {
                "speaker": "agent",
                "text": "Then the unit is faulty and I am not going to keep you chasing it.",
            },
            {
                "speaker": "agent",
                "text": (
                    "We tried a firmware update first. That is now cancelled; a replacement "
                    "pair takes its place."
                ),
            },
            {
                "speaker": "agent",
                "text": (
                    "That resolves the dropout fault. New set posted today under the "
                    "six-month accessory warranty, keep the old ones, no return needed."
                ),
            },
            {
                "speaker": "customer",
                "text": (
                    "that works. I half expected to be told to go and switch to Solent Audio "
                    "instead, which is what the forum kept telling me to do."
                ),
            },
            {
                "speaker": "agent",
                "text": (
                    "Not today. The replacement is booked against order B40877 and tracking "
                    "follows within the hour."
                ),
            },
            {
                "speaker": "customer",
                "text": "appreciated. that is the fastest anyone has ever sorted anything for me.",
            },
        ],
    },
    {
        "thread_id": "t03",
        "opened": "2026-03-18",
        "channel": "phone log",
        "subject": "N600 ordered on the tenth, still not shipped",
        "turns": [
            {
                "speaker": "customer",
                "text": (
                    "I ordered a Northwind Router N600 on the tenth. The site promised 48-hour "
                    "dispatch. It is the eighteenth and I have nothing."
                ),
            },
            {
                "speaker": "agent",
                "text": (
                    "Dev Shah, I have the account open — Ana Duarte, order C-51120, one N600, "
                    "business address in Porto. It shows as picked but never handed to the "
                    "carrier. Call logged as a late dispatch complaint, reported by Ana "
                    "Duarte, ref CS-10310."
                ),
            },
            {"speaker": "customer", "text": "Picked but not shipped. For eight days."},
            {
                "speaker": "agent",
                "text": (
                    "That is right, and it is not what we promised you. The 48-hour dispatch "
                    "commitment is missed by six days on this order."
                ),
            },
            {
                "speaker": "customer",
                "text": (
                    "I could have bought the same box from your competitor down the road, "
                    "Blackmere Direct, and carried it out the same afternoon. I stayed with "
                    "you because we run four sites on your kit."
                ),
            },
            {
                "speaker": "agent",
                "text": (
                    "I know, and the Business Plan account should have made this faster rather "
                    "than slower. Here is what I can do today: cancel the stuck pick, ship a "
                    "replacement N600 off the Lisbon shelf on an overnight service, and it is "
                    "with you before noon tomorrow."
                ),
            },
            {"speaker": "customer", "text": "Overnight is fine. Do not charge me for it."},
            {
                "speaker": "agent",
                "text": (
                    "I will not. I am also putting a €35 goodwill credit on the account for "
                    "the missed dispatch — that is separate from the price of the router, it "
                    "is not a discount on it."
                ),
            },
            {
                "speaker": "customer",
                "text": (
                    "Understood. Send me the tracking when it moves. And it goes to the Porto "
                    "site, not Lisbon — we closed Lisbon in February."
                ),
            },
            {
                "speaker": "agent",
                "text": (
                    "Porto is already the address on the order, so there is nothing to change. "
                    "Tracking to your email inside the hour."
                ),
            },
            {"speaker": "customer", "text": "Fine. It is the second time this year, though."},
            {
                "speaker": "agent",
                "text": (
                    "It is, and I have put a note on the account so that the next person sees "
                    "it without you having to say it again on the phone."
                ),
            },
        ],
    },
    {
        "thread_id": "t04",
        "opened": "2026-04-02",
        "channel": "email",
        "subject": "Charged twice for Priority Care on 28 March",
        "turns": [
            {
                "speaker": "customer",
                "text": (
                    "There are two charges on my card for Priority Care, both dated 28 March, "
                    "both $149. I have one account. Invoice D-77401 is the one I recognise; "
                    "the second has no invoice I can find anywhere in the portal.\n\n"
                    "Marcus Webb"
                ),
            },
            {
                "speaker": "agent",
                "text": (
                    "Hi Marcus, Dev Shah here. I can see both of them, and the second one is a "
                    "real capture rather than an authorisation hold. Your card was re-tokenised "
                    "on the twenty-seventh and the old token was never retired, so the annual "
                    "renewal fired twice against the same plan."
                ),
            },
            {
                "speaker": "customer",
                "text": "So your billing system charged me twice for the same year of the same plan.",
            },
            {
                "speaker": "agent",
                "text": (
                    "It did. That is ours and not yours. I am refunding the duplicate in full "
                    "— $149 back to the same card — and I have retired the stale token so it "
                    "cannot repeat next March."
                ),
            },
            {
                "speaker": "customer",
                "text": (
                    "Fine. While we are here: I never agreed to automatic renewal. I bought "
                    "Priority Care once, in 2024, and I expected it to lapse."
                ),
            },
            {
                "speaker": "agent",
                "text": (
                    "The annual auto-renewal clause is in the Priority Care terms accepted at "
                    "purchase, and a reminder goes out fourteen days ahead of the charge. I can "
                    "see it was sent on the fourteenth to this same address."
                ),
            },
            {
                "speaker": "customer",
                "text": (
                    "I do not doubt that you sent it. I am saying the clause is buried and the "
                    "reminder reads like marketing. Two people on my team were caught by the "
                    "same thing this year."
                ),
            },
            {
                "speaker": "agent",
                "text": (
                    "That is fair, and I have passed it to the billing product team in your "
                    "words rather than mine. On your account specifically I have switched "
                    "auto-renewal off, so the 2027 term will need you to say yes."
                ),
            },
            {"speaker": "customer", "text": "Then we are square. How long for the $149?"},
            {
                "speaker": "agent",
                "text": (
                    "Three to five working days under reference RF-3390. If it is not there on "
                    "the fifth working day, reply to this email and I will raise it as a "
                    "card-network trace instead of a refund."
                ),
            },
            {"speaker": "customer", "text": "Noted. Thank you for not making me argue about it."},
        ],
    },
    {
        "thread_id": "t05",
        "opened": "2026-04-21",
        "channel": "chat",
        "subject": "replacement N600 is dead, want the money back",
        "turns": [
            {
                "speaker": "customer",
                "text": (
                    "The N600 you shipped me in March is dead. It rebooted itself three times "
                    "last week and this morning it did not come back at all — no lights, "
                    "nothing. I am on chat rather than the phone because I have no patience "
                    "left for hold music."
                ),
            },
            {
                "speaker": "agent",
                "text": (
                    "Hi Ana, Tomas here. That is a bad one and I am sorry. I have opened "
                    "CS-10402 for it — a total power failure, reported by Ana Duarte. Before "
                    "I write the unit off: is the power brick warm to the touch, and does the "
                    "power LED do anything at all if you hold the reset pin for thirty "
                    "seconds?"
                ),
            },
            {
                "speaker": "customer",
                "text": (
                    "The brick is stone cold. I held the pin for a full minute and the LED "
                    "never lit."
                ),
            },
            {
                "speaker": "agent",
                "text": (
                    "A cold brick with no LED at all is the power supply rather than the router "
                    "board. Either way it is a warranty replacement — order C-51120 is inside "
                    "the two-year hardware warranty."
                ),
            },
            {"speaker": "customer", "text": "You said warranty replacement in March as well."},
            {
                "speaker": "agent",
                "text": (
                    "I know. This one would be the replacement for a replacement, and I "
                    "understand exactly how that reads from your side. What I can do right now "
                    "is ship a new N600 today on an overnight service and put a spare power "
                    "supply in the box, so that a dead brick never takes the site down again."
                ),
            },
            {
                "speaker": "customer",
                "text": (
                    "No. It would be the third one in three months. We have already switched "
                    "the office to Cobalt Networks and their box has not dropped once in nine "
                    "days. I do not want another router, I want the money back."
                ),
            },
            {
                "speaker": "agent",
                "text": "Then we do that instead.",
            },
            {
                "speaker": "agent",
                "text": (
                    "I booked an overnight replacement a minute ago. That is now cancelled; a "
                    "full refund takes its place."
                ),
            },
            {
                "speaker": "agent",
                "text": (
                    "€289 back to the company card, on order C-51120. That resolves the total "
                    "power failure and closes the warranty claim on the router."
                ),
            },
            {"speaker": "customer", "text": "And the €35 credit from March?"},
            {
                "speaker": "agent",
                "text": (
                    "It stays exactly where it is. It was goodwill for the missed dispatch, not "
                    "part of what you paid for the router, so the refund does not claw it back."
                ),
            },
            {"speaker": "customer", "text": "Good. Send me something in writing."},
            {
                "speaker": "agent",
                "text": (
                    "Confirmation is going to the Porto address on the account under reference "
                    "RF-4118, and I am leaving this chat transcript on the case so that nobody "
                    "asks you to explain it again."
                ),
            },
        ],
    },
    {
        "thread_id": "t06",
        "opened": "2026-05-06",
        "channel": "email",
        "subject": "Fourth request: refund RF-2261 has still not been paid",
        "turns": [
            {
                "speaker": "customer",
                "text": (
                    "This is the fourth email I have written about order #A-99213. The refund "
                    "was authorised on 24 March under reference RF-2261. It is now 6 May. The "
                    "money is not in the account and nobody has told me why."
                ),
            },
            {
                "speaker": "agent",
                "text": (
                    "Priya — I am sorry, and I can see exactly what happened. RF-2261 was "
                    "authorised and then dropped into a manual review queue because the card it "
                    "was raised against had been replaced. Nobody actioned the queue."
                ),
            },
            {
                "speaker": "customer",
                "text": (
                    "Somebody called me on 21 April and promised a callback inside a day with "
                    "an answer. It never came. That is the second time a 24-hour callback "
                    "commitment has been made to me on this case and missed."
                ),
            },
            {
                "speaker": "agent",
                "text": (
                    "Both of those are on us. I have re-issued the refund against the "
                    "replacement card as RF-2261-B, and I am escalating this to Tomas Ferreira, "
                    "who runs the refunds bench, so that somebody senior owns it instead of me "
                    "chasing it."
                ),
            },
            {
                "speaker": "customer",
                "text": (
                    "I do not want it watched. I want it paid. My finance director has asked me "
                    "twice this month why a laptop we returned in March is still sitting on the "
                    "ledger."
                ),
            },
            {
                "speaker": "agent",
                "text": (
                    "Priya, this is Tomas Ferreira — Dev handed this to me an hour ago. I have "
                    "pulled RF-2261-B out of the review queue by hand and it is with the card "
                    "network now. I am not going to give you a date I cannot hold, so instead: I "
                    "will email you every working day until it clears, whether or not there is "
                    "news."
                ),
            },
            {
                "speaker": "customer",
                "text": (
                    "That is the first useful sentence anyone has written to me in six weeks. We "
                    "were a fourteen-machine Aurora shop. We have switched the fleet to your "
                    "competitor Kestrel Compute; the order went in on Friday and I signed it "
                    "myself."
                ),
            },
            {
                "speaker": "agent",
                "text": (
                    "I understand, and I am not going to try to talk you out of it while your "
                    "money is still sitting with us."
                ),
            },
            {
                "speaker": "customer",
                "text": (
                    "The Aurora is still boxed in my hallway, incidentally. Nobody ever sent a "
                    "courier for it."
                ),
            },
            {
                "speaker": "agent",
                "text": (
                    "That is the fourth thing to have gone wrong on this case. I have booked the "
                    "collection for Thursday and I will confirm the reference tonight. This stays "
                    "open under my name until the refund clears and the machine is gone."
                ),
            },
        ],
    },
    {
        "thread_id": "t07",
        "opened": "2026-05-19",
        "channel": "chat",
        "subject": "cancel Priority Care and order F-88132",
        "turns": [
            {
                "speaker": "customer",
                "text": (
                    "two things to cancel, and chat is quicker than the phone: the Priority "
                    "Care plan, and order F-88132 — the second pair of Halcyon Buds I ordered "
                    "on Saturday. Nothing is wrong with either, I just do not need them."
                ),
            },
            {
                "speaker": "agent",
                "text": (
                    "Hi Marcus, Tomas here. Both are straightforward. F-88132 has not been "
                    "picked yet, so I can pull it before it ships. Give me a moment."
                ),
            },
            {
                "speaker": "agent",
                "text": (
                    "Done. F-88132 is cancelled and the authorisation drops off your card in "
                    "three to five days — it was never captured, so there is nothing to refund "
                    "on that one."
                ),
            },
            {"speaker": "customer", "text": "Good. And the plan?"},
            {
                "speaker": "agent",
                "text": (
                    "Priority Care renewed on 28 March and runs to 27 March 2027. You are "
                    "outside the 14-day cooling-off period, so strictly it is non-refundable "
                    "once the term has started."
                ),
            },
            {"speaker": "customer", "text": "that is not what the last agent told me in April."},
            {
                "speaker": "agent",
                "text": (
                    "I have read that thread. Dev switched your auto-renewal off after the "
                    "duplicate charge, and given that history I am not going to make you argue "
                    "this one. I am cancelling the plan today and issuing a pro-rata credit of "
                    "$61 for the ten unused months."
                ),
            },
            {
                "speaker": "customer",
                "text": "I will take that. does cancelling the plan touch the warranty on the Buds?",
            },
            {
                "speaker": "agent",
                "text": (
                    "No. The six-month accessory warranty rides with the product rather than "
                    "with the plan, so the replacement pair from March is covered until "
                    "September either way."
                ),
            },
            {"speaker": "customer", "text": "that is everything then. thanks."},
            {
                "speaker": "agent",
                "text": (
                    "Cancellation confirmed under reference RF-5027. The credit is on its way "
                    "and the account drops to standard cover from today."
                ),
            },
        ],
    },
    {
        "thread_id": "t08",
        "opened": "2026-06-08",
        "channel": "chat",
        "subject": "charging case only charges the right earbud",
        "turns": [
            {
                "speaker": "customer",
                "text": (
                    "Small one. The charging case for my Halcyon Buds only charges the right "
                    "earbud. The left one comes out at whatever percentage it went in at."
                ),
            },
            {
                "speaker": "agent",
                "text": (
                    "Hi Ana, Dev Shah. Opening CS-10477: a charging fault, reported by Ana "
                    "Duarte. Does the left bud seat all the way — does it click, and does the "
                    "case LED blink twice when you shut the lid?"
                ),
            },
            {
                "speaker": "customer",
                "text": "It clicks. The LED does one long blink instead of two.",
            },
            {
                "speaker": "agent",
                "text": (
                    "One long blink means the case can see a bud but is not making a charge "
                    "circuit. Nine times in ten that is the contact pins in the left well "
                    "rather than the earbud itself. Have a look with a torch: do the two pins "
                    "on the left sit at the same height as the two on the right?"
                ),
            },
            {
                "speaker": "customer",
                "text": "One of the left pins is flush with the plastic. On the right they both stand proud.",
            },
            {
                "speaker": "agent",
                "text": (
                    "A collapsed pin, then, and that is a fault in the charging case rather "
                    "than in the earphones. I am raising an RMA on the case alone under the "
                    "six-month accessory warranty, against order H-60244 — keep the earphones, "
                    "post back the case only."
                ),
            },
            {"speaker": "customer", "text": "How long am I without it?"},
            {
                "speaker": "agent",
                "text": (
                    "You are covered by the five working day RMA turnaround, counted from the "
                    "day the case reaches the returns bench."
                ),
            },
            {
                "speaker": "customer",
                "text": (
                    "It has been eleven days. I posted it on the ninth and the portal still "
                    "says awaiting receipt."
                ),
            },
            {
                "speaker": "agent",
                "text": (
                    "I can see it was signed for on the eleventh and then never booked in. I "
                    "have asked the returns bench to book it in by hand today, and I am not "
                    "closing this until you have a working case back."
                ),
            },
            {"speaker": "customer", "text": "Please do not close it. This is the third thing this year."},
            {
                "speaker": "agent",
                "text": (
                    "It stays open under my name, and I have left this chat transcript on the "
                    "case so that the next person does not start from nothing."
                ),
            },
        ],
    },
]


# The hand-written answer key. Canonical names, not surface forms: "A-99213"
# stands for "#A-99213" and "order A99213" alike, "Aurora 14" for "the Aurora"
# and "Aurora-14s". Curated rather than exhaustive -- every relation in
# CUSTOMER_SERVICE appears at least twice, and the three `replaces` edges are
# the supersession key for threads t01, t02 and t05.
GOLD_CS_FACTS: list[tuple[str, str, str]] = [
    # --- who raised it, who worked it -------------------------------------
    ("Aurora 14 shuts down at 40% charge", "reported_by", "Priya Raman"),
    ("left Halcyon Bud drops out", "reported_by", "Marcus Webb"),
    ("charging case will not charge left earbud", "reported_by", "Ana Duarte"),
    ("Aurora 14 shuts down at 40% charge", "handled_by", "Dev Shah"),
    ("left Halcyon Bud drops out", "handled_by", "Tomas Ferreira"),
    ("refund RF-2261 never paid", "escalated_to", "Tomas Ferreira"),
    # --- what the case is about -------------------------------------------
    ("Aurora 14 shuts down at 40% charge", "concerns_product", "Aurora 14"),
    ("N600 dead, no power", "concerns_product", "Northwind Router N600"),
    ("Aurora 14 shuts down at 40% charge", "about_order", "A-99213"),
    ("left Halcyon Bud drops out", "about_order", "B-40877"),
    ("N600 dead, no power", "about_order", "C-51120"),
    ("A-99213", "owned_by", "Priya Raman"),
    ("C-51120", "owned_by", "Ana Duarte"),
    # --- root causes -------------------------------------------------------
    ("Aurora 14 shuts down at 40% charge", "caused_by", "battery"),
    ("N600 dead, no power", "caused_by", "power supply"),
    ("duplicate Priority Care charge", "caused_by", "annual auto-renewal clause"),
    ("battery", "component_of", "Aurora 14"),
    ("left earbud", "component_of", "Halcyon Buds"),
    ("contact pins", "component_of", "charging case"),
    # --- remedies, and remedies that replaced earlier remedies -------------
    ("Aurora 14 shuts down at 40% charge", "resolved_by", "full refund on A-99213"),
    ("left Halcyon Bud drops out", "resolved_by", "replacement pair of Halcyon Buds"),
    ("N600 dead, no power", "resolved_by", "full refund on C-51120"),
    ("full refund on A-99213", "replaces", "advance replacement Aurora 14"),
    ("replacement pair of Halcyon Buds", "replaces", "firmware 2.1.6 update"),
    ("full refund on C-51120", "replaces", "overnight replacement N600"),
    ("Priya Raman", "compensated_with", "£1,249 refund"),
    ("Ana Duarte", "compensated_with", "€35 goodwill credit"),
    ("Marcus Webb", "compensated_with", "$61 pro-rata credit"),
    # --- the rules invoked, and the clocks missed --------------------------
    ("Aurora 14 shuts down at 40% charge", "refers_to_policy", "12-month limited warranty"),
    ("cancellation of order F-88132", "refers_to_policy", "14-day cooling-off period"),
    ("Aurora 14 shuts down at 40% charge", "breaches", "two-business-day depot turnaround"),
    ("N600 order not dispatched", "breaches", "48-hour dispatch"),
    ("charging case will not charge left earbud", "breaches", "five working day RMA turnaround"),
    # --- account context ---------------------------------------------------
    ("Priya Raman", "has_tier", "Priority Care"),
    ("Ana Duarte", "has_tier", "Business Plan"),
    ("Priya Raman", "mentions_competitor", "Kestrel Compute"),
    ("Ana Duarte", "mentions_competitor", "Cobalt Networks"),
    ("Priya Raman", "contacted_via", "email"),
    ("Marcus Webb", "contacted_via", "chat"),
    ("Ana Duarte", "contacted_via", "chat"),
]


# Thread-level labels. None of these are spans in the text -- nobody writes
# "priority: urgent" -- so they need a classification pass rather than an
# extraction pass. `intent` is what the customer wanted when they opened the
# thread, NOT how it ended: t01 opens as technical support and ends in a refund,
# and labelling it "refund request" is the mistake to watch for.
GOLD_INTENTS: dict[str, dict] = {
    "t01": {"intent": "technical support", "priority": "high", "resolved": True},
    "t02": {"intent": "technical support", "priority": "medium", "resolved": True},
    "t03": {"intent": "order status", "priority": "low", "resolved": True},
    "t04": {"intent": "billing dispute", "priority": "medium", "resolved": True},
    "t05": {"intent": "refund request", "priority": "high", "resolved": True},
    "t06": {"intent": "complaint", "priority": "urgent", "resolved": False},
    "t07": {"intent": "cancellation", "priority": "medium", "resolved": True},
    "t08": {"intent": "technical support", "priority": "low", "resolved": False},
}

INTENT_LABELS: tuple[str, ...] = (
    "refund request",
    "technical support",
    "order status",
    "billing dispute",
    "complaint",
    "cancellation",
)

PRIORITY_LABELS: tuple[str, ...] = ("low", "medium", "high", "urgent")


# Gold clustering for entity resolution (B-cubed). canonical -> every surface
# form that occurs literally in some turn of THREADS. Verified by the test at
# the bottom of notebook 09; if you edit a turn, re-run it.
ALIAS_GROUPS: dict[str, list[str]] = {
    "Aurora 14": [
        "Aurora 14",
        "Aurora-14s",
        "The Aurora",
        "Aurora shop",
        "the laptop",
        "a laptop",
    ],
    "Halcyon Buds": [
        "Halcyon Buds",
        "the Buds",
        "the earphones",
    ],
    "Northwind Router N600": [
        "Northwind Router N600",
        "The N600",
        "N600",
        "the router",
        "another router",
    ],
    "A-99213": [
        "order A-99213",
        "#A-99213",
        "order A99213",
        "A-99213",
    ],
    "B-40877": [
        "#B-40877",
        "order B40877",
        "B-40877",
    ],
    "C-51120": [
        "order C-51120",
        "C-51120",
    ],
    "Priya Raman": [
        "Priya Raman",
        "Priya",
    ],
    "Marcus Webb": [
        "Marcus Webb",
        "Marcus",
    ],
    "Ana Duarte": [
        "Ana Duarte",
        "Ana",
    ],
    "Dev Shah": [
        "Dev Shah",
        "Dev",
    ],
    "Tomas Ferreira": [
        "Tomas Ferreira",
        "Tomas",
    ],
    "Priority Care": [
        "Priority Care",
        "the Priority Care plan",
        "the plan",
    ],
    "RF-2261": [
        "RF-2261",
        "RF-2261-B",
    ],
}


# Turns whose fact is unrecoverable without coreference: the subject is a bare
# pronoun or a definite description and the antecedent is in an EARLIER turn,
# usually spoken by the other party. `turn` indexes THREADS[i]["turns"].
#
# This is the measurable claim the domain exists to support: run the pipeline
# over raw turns, then over turns rewritten by
# kgx.coref.ConversationPreprocessor, and count how many of these facts land.
PRONOUN_TARGETS: list[dict] = [
    {
        "thread_id": "t01",
        "turn": 6,
        "mention": "It is nine business days later ... I posted it the morning after we spoke",
        "antecedent": "Aurora 14",
        "fact": "the laptop was posted to the depot and never booked in",
    },
    {
        "thread_id": "t02",
        "turn": 2,
        "mention": "it comes back after about ten seconds on its own",
        "antecedent": "left earbud",
        "fact": "the left earbud drops out and self-recovers",
    },
    {
        "thread_id": "t02",
        "turn": 6,
        "mention": "it's on 2.1.6 now and it still cuts out",
        "antecedent": "Halcyon Buds",
        "fact": "firmware 2.1.6 did not fix the fault -- this is what makes the replacement supersede it",
    },
    {
        "thread_id": "t03",
        "turn": 8,
        "mention": "And it goes to the Porto site, not Lisbon",
        "antecedent": "overnight replacement N600",
        "fact": "the replacement router ships to Porto",
    },
    {
        "thread_id": "t04",
        "turn": 3,
        "mention": "It did.",
        "antecedent": "the billing system charged Marcus Webb twice",
        "fact": "the agent confirms the duplicate charge -- the whole fact is one pronoun and a verb",
    },
    {
        "thread_id": "t05",
        "turn": 6,
        "mention": "It would be the third one in three months.",
        "antecedent": "the replacement N600 offered in the previous turn",
        "fact": "the customer refuses a third router, which is what turns the replacement into a refund",
    },
    {
        "thread_id": "t05",
        "turn": 11,
        "mention": "It stays exactly where it is. It was goodwill for the missed dispatch",
        "antecedent": "€35 goodwill credit",
        "fact": "the March credit survives the refund and is not part of the price",
    },
    {
        "thread_id": "t06",
        "turn": 4,
        "mention": "I do not want it watched. I want it paid.",
        "antecedent": "refund RF-2261-B",
        "fact": "the re-issued refund is still unpaid",
    },
    {
        "thread_id": "t07",
        "turn": 4,
        "mention": "so strictly it is non-refundable once the term has started",
        "antecedent": "the Priority Care plan",
        "fact": "the plan is non-refundable under the 14-day cooling-off period",
    },
    {
        "thread_id": "t08",
        "turn": 2,
        "mention": "It clicks. The LED does one long blink instead of two.",
        "antecedent": "the left earbud",
        "fact": "the left bud seats but draws no charge -- the diagnostic that identifies the contact pins",
    },
    {
        "thread_id": "t08",
        "turn": 8,
        "mention": "It has been eleven days. I posted it on the ninth",
        "antecedent": "charging case",
        "fact": "the case was posted and the RMA clock has been missed",
    },
]
