"""Domain ontologies for notebook 09.

:mod:`kgx.ontology` ships two ontologies -- ``AGENT_MEMORY`` and
``BUSINESS_NEWS`` -- because the earlier notebooks need exactly those two. This
module holds the rest: one ontology per domain that notebook 09 sweeps over, so
that "does this pipeline transfer to a new domain?" is answered by importing a
different constant rather than by rewriting the pipeline.

Every ontology here obeys the same two hard limits as the shipped pair. Roughly
13 entity types and 15-16 relations, because each type name and description is
serialised into the encoder input on every call and a bigger vocabulary costs
recall measurably. And no relation sets ``symmetric=True``: in ``gliner2``
2.0.0 that flag compiles to a constraint set which rejects every candidate edge,
so the relation silently yields zero results. Use a directed relation, or a pair
joined by ``inverse=``.

Each ontology is written against a specific corpus in :mod:`kgx.data`, and every
relation it declares is one that corpus actually expresses. Declaring a relation
the text never states is not free -- it burns encoder budget and produces an
ontology diagram that lies about the graph you can build.
"""

from .ontology import EntityType, Ontology, RelationType


# ---------------------------------------------------------------------------
# Shopping / ecommerce -- product catalogue, reviews and aspect-level opinion
# ---------------------------------------------------------------------------
# The corpus is kgx.data.shopping. Two decisions are worth arguing with:
#
#   1. `product_aspect` and `specification` are separate node types even though
#      both look like "a property of the product". They behave completely
#      differently: "68 Wh battery" is a fact from a spec table that belongs to
#      the catalogue, and "battery life" is a dimension a buyer judges. Collapsed
#      into one type, the aspect-sentiment pass has to sift specs back out, and
#      the catalogue picks up "battery life: bad" as an attribute.
#   2. There is no `positive_aspect` / `negative_aspect` split, and no sentiment
#      node. Polarity is one attribute of a `has_aspect` edge; JointSchema edges
#      carry no attributes, so it is recovered by the span-attribute pass in
#      kgx.extract and scored against GOLD_ASPECTS. Splitting the predicate
#      instead would double the relation budget for one bit of information.
#
# `replaces` is the trap this domain exists for. Aurora 14 -> Aurora 14 Pro and
# Halcyon Buds -> Halcyon Buds Pro look exactly like successor pairs and are not
# (both models stay in the range); Northwind Router N600 -> N600X is. The
# description has to carry that distinction, because the surface strings cannot.

SHOPPING = Ontology(
    name="shopping",
    description=(
        "Ecommerce catalogue and review graph: which brand makes what, who sells "
        "it at what price under which offer, what fits what, and which aspect of "
        "a product a buyer is actually praising or attacking."
    ),
    entities=(
        EntityType("product", "A specific purchasable model, at model-number granularity: 'Aurora 14 Pro', 'Northwind Router N600X'. The trim or suffix is part of the identity - 'Aurora 14' and 'Aurora 14 Pro' are two products, not one. Not the brand on its own, not a category."),
        EntityType("brand", "The manufacturer or label a product is sold under: Aurora, Halcyon, Northwind. Not the shop that takes the order - that is a retailer."),
        EntityType("product_category", "A shelf, department or product class: ultrabooks, true wireless earbuds, mesh routers, docking stations. Always a class of things, never one model."),
        EntityType("retailer", "A shop, marketplace or seller that fulfils the order: Vellum Market, Kestrel Electronics. Not the manufacturer, even when a brand runs its own storefront."),
        EntityType("product_aspect", "The facet of a product or seller that an opinion is about: battery life, build quality, screen, keyboard, fit, noise cancellation, price, shipping, returns, customer support. The dimension being judged - not the judgement itself and not the specific defect."),
        EntityType("price", "A money amount asked or paid: $1,299, $76.49, $59. Not a discount percentage and not the name of the sale."),
        EntityType("promotion", "A named sale, coupon, bundle deal or trade-in offer: Back-to-Campus promotion, Vellum Autumn Days, Clearance Blowout, a $25 trade-in credit. Not the price that results from it."),
        EntityType("review_author", "The person whose opinion the text reports: a reviewer's name, handle, or 'verified buyer'. Not a support agent answering a buyer's question."),
        EntityType("specification", "A published, measurable attribute: 68 Wh battery, tri-band Wi-Fi 7, IPX4, 1 TB SSD, 45 W charger, 2.5 GbE WAN port. A number from a spec table, not an opinion about it - '68 Wh battery' is a specification, 'battery life' is a product_aspect."),
        EntityType("accessory", "An add-on bought for a product rather than the product itself: a dock, charging case, ear tips, wall mount kit, travel charger, cable."),
        EntityType("complaint", "A concrete stated defect or bad experience: 'the left bud drops out', 'firmware 2.1.4 bricked one of my nodes', 'a bent HDMI port', 'the phone queue ran to 35 minutes'. Usually a clause with a verb, describing something that happened. A bare noun phrase naming a dimension is a product_aspect instead - 'fan noise' is an aspect, 'the fans spin up loudly during any compile' is a complaint."),
        EntityType("shipping_option", "A named delivery or collection method and its terms: free two-day delivery, free standard shipping, next-day courier, in-store pickup."),
        EntityType("warranty", "A cover or protection term: two-year limited warranty, 12-month manufacturer warranty, Aurora Care+ extended plan. The length is part of the identity - a 12-month and a 24-month warranty are different things."),
    ),
    relations=(
        RelationType("made_by", ("product", "accessory"), ("brand",),
                     "The tail brand manufactures or badges the head. One brand per product, "
                     "so this is unique_head.", unique_head=True),
        RelationType("sold_by", ("product", "accessory"), ("retailer",),
                     "The tail retailer takes the order for the head. Deliberately NOT unique: "
                     "the same model is legitimately listed by several retailers at once."),
        RelationType("belongs_to_category", ("product", "accessory"), ("product_category",),
                     "The head is an instance of the tail shelf or department. Not brand "
                     "membership and not a bundle."),
        RelationType("has_aspect", ("product", "accessory", "retailer"), ("product_aspect",),
                     "The text passes judgement on this facet of the head. Attach the aspect to "
                     "whatever is actually being judged - shipping and returns attach to the "
                     "retailer, battery life to the product. Polarity is not part of this edge; "
                     "it is recovered separately as a span attribute."),
        RelationType("priced_at", ("product", "accessory"), ("price",),
                     "The head is offered at, or was bought for, the tail amount. Not unique: "
                     "list price, sale price and what one buyer paid can all be stated."),
        RelationType("compatible_with", ("accessory", "product"), ("product", "accessory"),
                     "The head is stated to work with the tail. Only for an asserted fit - a "
                     "sentence saying the head will NOT fit or mesh with the tail is not this "
                     "relation."),
        RelationType("compares_to", ("product",), ("product",),
                     "The text weighs the head against the tail: a buyer upgrading between them, "
                     "a listing telling shoppers which of the two to pick. Not a successor "
                     "claim - that is replaces."),
        RelationType("complains_about", ("review_author", "complaint"),
                     ("product", "accessory", "product_aspect", "retailer", "shipping_option"),
                     "The head is unhappy with the tail, or is a defect occurring in it. Use it "
                     "for 'I would not recommend X' as well as for a named fault."),
        RelationType("recommends", ("review_author",), ("product", "accessory", "retailer"),
                     "The head endorses the tail to other buyers. Positive endorsement only; a "
                     "negative verdict is complains_about, not a recommends edge."),
        RelationType("has_specification", ("product", "accessory"), ("specification",),
                     "A published measurable attribute of the head. Catalogue data, not opinion."),
        RelationType("bundled_with", ("product",), ("accessory", "product"),
                     "The head is the thing being bought; the tail comes in the same box or the "
                     "same purchase. Direction matters - the laptop is bundled with the dock, "
                     "not the other way round. Not compatible_with: an accessory that merely "
                     "works with a product is not bundled with it."),
        RelationType("replaces", ("product",), ("product",),
                     "The head is the successor model that supersedes the tail in the range. A "
                     "pricier trim sold alongside the tail is NOT a replacement, however similar "
                     "the names look - check that the tail is actually discontinued or outgoing.",
                     acyclic=True),
        RelationType("covered_by", ("product", "accessory"), ("warranty",),
                     "The head is protected by this warranty or cover plan for the stated term."),
        RelationType("discounted_by", ("product", "accessory"), ("promotion",),
                     "The head's price is cut by this named offer."),
        # threshold=0.35 is measured, not taste. In an isolated sentence this edge
        # decodes at 0.6-0.8, but in a full 190-word listing it is crowded out by
        # the denser priced_at / discounted_by / covered_by edges competing for the
        # same product node and never appears at the default threshold. Shipping
        # terms are exactly the kind of low-frequency, high-value fact a catalogue
        # graph is built to answer, so the relation gets its own floor.
        RelationType("ships_with", ("product", "accessory", "retailer"), ("shipping_option",),
                     "The head is delivered or collected by this method on the stated terms. "
                     "Attach it to the product when a listing quotes the term for one item, and "
                     "to the retailer when the term is a store-wide policy.",
                     threshold=0.35),
        RelationType("offered_by", ("promotion", "warranty", "shipping_option"),
                     ("retailer", "brand"),
                     "The tail runs, honours or funds the head. Use it when the text says who "
                     "stands behind an offer, a cover plan or a delivery term - which matters "
                     "when only one seller of a product honours it."),
    ),
)


# ---------------------------------------------------------------------------
# Travel -- bookings, flights, stays, disruptions
# ---------------------------------------------------------------------------
# Corpus: kgx.data.travel.
#
# The distinguishing problem in this domain is not ambiguity, it is *controlled
# vocabulary*. Half the entities are codes -- LHR, SIN, MR, NW, MR117,
# MR7K2QX9, SKY-4471902 -- and "LHR" and "London Heathrow" share no characters,
# so neither edit distance nor an embedding will ever merge them. The extractor
# only has to find the span and type it; the merge is a dictionary lookup
# against kgx.data.travel.IATA_GAZETTEER. Type descriptions below are therefore
# written to help GLiNER separate code *namespaces* that look alike (a flight
# number, a booking locator and a loyalty number are all letter-digit blobs),
# because that is the distinction a downstream gazetteer cannot repair.
#
# `departs_from` / `arrives_at` are kept apart rather than folded into one
# `route` edge with a role attribute, because JointSchema edges carry no
# properties and origin-vs-destination is the whole content of the fact. Both
# are unique_head: a leg has exactly one origin and one destination, which also
# stops the decoder from attaching a connection airport to both ends.
#
# `replaces` is the edge that makes a rebooking legible: the new ticket points
# at the old one, so the original stays in the graph as a disrupted node rather
# than being overwritten. Compare AGENT_MEMORY.replaces, which does the same job
# for tool migrations.

#
# One measured negative result worth keeping, because it is the ontology-size
# tradeoff made concrete. `covers` (booking -> segment) is expressed in the
# corpus in plain words -- "the reference covers MR117 and MR118", "Record
# locator NWQ9ZP41 covers NW612, NW613 and a Narita Express ticket" -- and in an
# isolated 4-type / 1-relation subset GLiNER2.5 recovers it 19 times with the
# right endpoints. Inside this full 13-type / 16-relation ontology it fires
# once. Nothing is wrong with the relation or the text; the relation-count head
# simply spends its budget elsewhere. Renaming it does not help either: a sweep
# over ten labels (includes, contains, comprises, ticketed_as, paid_for, ...)
# moved the count but not the correctness. If you need the booking-to-segment
# edge reliably, recover it from document structure -- a locator header followed
# by a schedule block -- rather than asking the model for it, or run it as its
# own narrow pass via Ontology.subset().

TRAVEL = Ontology(
    name="travel",
    description=(
        "Trip records as airlines, hotels and expense systems actually write "
        "them: who flew where on whose metal, under which locator, staying "
        "where, and what went wrong. Half the graph is codes."
    ),
    entities=(
        EntityType("traveller", "The person flying or staying - the passenger or guest named on a booking. Not the agent, desk or approver who issued it."),
        EntityType("airline", "A carrier, by name or by its two-letter code (MR, NW). Not the aircraft type, not the airport, and not the loyalty programme it runs."),
        EntityType("flight", "One scheduled leg, almost always a flight number: MR117, NW612, 'Meridian Air 117'. Not the whole trip - a multi-leg itinerary is a booking."),
        EntityType("airport", "A named airport or its three-letter IATA code: LHR, Singapore Changi, NRT. Not the city it serves, even when the two share a name."),
        EntityType("city", "A city or metropolitan area as a place people go to: London, Tokyo, New York. Not the airport code standing in for it."),
        EntityType("country", "A sovereign country or territory: Japan, Germany, the United Kingdom. Not a city and not a region."),
        EntityType("hotel", "A named property where a traveller sleeps. The property name, not the confirmation code for the stay."),
        EntityType("booking", "A reservation locator: PNR, booking reference, record locator, e-ticket number, hotel confirmation code. The identifier itself - MR7K2QX9, KRH-88214 - not the flight or room it buys."),
        EntityType("loyalty_program", "A named frequent-flyer or guest-rewards scheme, or a membership number in one: Meridian Skyline, SKY-4471902. Not the airline that operates it and not a booking reference."),
        EntityType("ground_transport", "A named non-air surface service: airport express train, transfer car, shuttle coach, hire car. Never a flight."),
        EntityType("travel_document", "Something that authorises travel or entry: passport, visa, arrival card, electronic travel authorisation. Not a boarding pass, ticket or receipt."),
        EntityType("trip_purpose", "Why the trip is happening: a named conference, a client onsite, an audit, a research visit, annual leave. Not the destination and not the employer."),
        EntityType("disruption", "The thing that went wrong: cancellation, delay, strike or walkout, storm, runway closure, ground stop, missed connection. Not the replacement flight that fixed it."),
    ),
    relations=(
        RelationType("booked", ("traveller",), ("booking",),
                     "The traveller is the passenger or guest named on this reservation record."),
        RelationType("covers", ("booking",), ("flight", "hotel", "ground_transport"),
                     "The reservation record includes this segment. Head is the locator, tail is the thing flown, slept in or ridden."),
        RelationType("flies_on", ("traveller",), ("flight",),
                     "The traveller is ticketed on this leg. Person-to-flight only; the locator-to-flight link is 'covers'."),
        RelationType("operated_by", ("flight", "loyalty_program"), ("airline",),
                     "The carrier that flies this leg or runs this loyalty scheme. Use it for the operating carrier even when the ticket was issued on another airline's stock.",
                     unique_head=True),
        RelationType("departs_from", ("flight",), ("airport",),
                     "Origin airport of this leg. Exactly one per flight - a connection point is not an origin.",
                     unique_head=True),
        RelationType("arrives_at", ("flight",), ("airport",),
                     "Destination airport of this leg. Exactly one per flight - a connection point is not a destination.",
                     unique_head=True),
        RelationType("connects_through", ("booking", "flight"), ("airport", "city"),
                     "A transfer, transit or layover point partway along the itinerary. Never the origin and never the final destination."),
        RelationType("stays_at", ("traveller",), ("hotel",),
                     "The traveller has a room at this property for at least one night, including a held or day room."),
        RelationType("located_in", ("airport", "hotel", "ground_transport", "city"), ("city", "country"),
                     "Where the head physically sits: an airport in a city, a hotel in a city, a city in a country. Containment, not service.",
                     acyclic=True),
        RelationType("member_of", ("traveller",), ("loyalty_program",),
                     "The traveller holds an account in this scheme, usually with a membership number and a tier. Not a one-off partner credit."),
        RelationType("serves", ("airline",), ("airport", "city"),
                     "The carrier operates scheduled service to or from this place. A route-network fact about the airline, not about one dated flight."),
        RelationType("travels_for", ("traveller", "booking"), ("trip_purpose",),
                     "The stated reason the trip is being made."),
        RelationType("holds_document", ("traveller",), ("travel_document",),
                     "The traveller possesses, has applied for, or must renew this document."),
        RelationType("required_for", ("travel_document",), ("country", "trip_purpose"),
                     "Entry to, or participation in, the tail is conditional on holding the head document."),
        RelationType("disrupted_by", ("flight", "booking", "ground_transport", "traveller"), ("disruption",),
                     "The head was cancelled, delayed, suspended or otherwise degraded by the tail event."),
        RelationType("replaces", ("flight", "booking"), ("flight", "booking"),
                     "The head was issued to supersede the tail: a rebooking, a reissued coupon, a protected segment. Head is the new one, tail is the original.",
                     acyclic=True),
    ),
)


# ---------------------------------------------------------------------------
# Customer service -- support threads, and what was promised versus done
# ---------------------------------------------------------------------------
# The corpus is kgx.data.customer_service. The domain's distinguishing problem
# is that a support thread has no narrator: the subject of most facts is a
# pronoun ("It did.", "It would be the third one in three months") whose
# antecedent is in an earlier turn, usually the other party's. Nothing in the
# ontology fixes that -- it is a coreference problem, and the ontology is only
# the thing that says which recovered facts are worth keeping. Run this against
# raw turns and against turns rewritten by kgx.coref.ConversationPreprocessor;
# the delta is the whole argument for the preprocessor.
#
# Three shapes here are deliberate:
#
#   1. `issue` and `resolution` are separate node types rather than one "case"
#      node with a status property. JointSchema nodes carry no attributes, and
#      the interesting edge -- `replaces`, one remedy superseding another -- has
#      to connect two remedies. Threads t01, t02 and t05 each offer a remedy and
#      then change it; a pipeline that stores both as live facts asserts that
#      the same laptop was both replaced and refunded.
#   2. `refund` is carved out of `resolution` because money moving back to the
#      customer is the thing anyone querying this graph actually asks about, and
#      because "£1,249" and "a replacement unit" are lexically nothing alike --
#      one label covering both measurably dilutes it.
#   3. Intent, priority and resolution state are NOT in the ontology. They are
#      properties of the thread rather than spans in it, nobody writes "priority:
#      urgent", and a span extractor cannot produce them. They come from a
#      separate classification pass scored against GOLD_INTENTS. This is the
#      honest version of the boundary; the tempting alternative is an
#      `intent` entity type that fires on nothing.
#   4. `contacted_via` is headed by the customer, not by the issue. The obvious
#      shape is `reported_via(issue -> channel)` and it was measured at exactly
#      zero edges on this corpus: nobody in a support thread writes "this fault
#      was reported via chat". What people do write is "I tried the phone line
#      and gave up, so chat it is" -- a first-person statement about the
#      customer. Same channel node, a head the text actually supplies.
#
# Two things were measured on gliner2.5-base over kgx.data.customer_service, and
# both are worth carrying into the notebook rather than hiding.
#
# Relations headed by `issue` are the fragile half of this ontology. A fault is
# stated as a clause ("it has been shutting off without warning") and GLiNER's
# span head wants a noun phrase, so `issue` spans come out short, ragged, or not
# at all. What rescues them is the agent nominalising the fault back into a noun
# ("logged as an unexpected shutdown fault, reported by Priya Raman"): with that
# restatement reported_by fires at 0.92-0.99, without it reported_by,
# resolved_by and breaches were all zero. Threads t04, t06 and t07 carry no such
# restatement on purpose, so the drop-off is measurable rather than assumed.
#
# The extraction unit is per relation, not global. Over 4-turn windows every
# relation here fires except `replaces`; over single turns `replaces` fires and
# about_order, breaches and resolved_by stop. `replaces` is the extreme case: it
# scores 0.63-0.76 on the bare sentence "That is now cancelled; a full refund
# takes its place" and collapses to zero once two turns of surrounding context
# are added, at any threshold -- the relation-count head decides there are no
# instances. Hence `threshold=0.15` on it, which is what recovers the one
# correct supersession edge (t05, 'full refund' -> 'overnight replacement',
# 0.49); at the default it yields nothing at all. Two remedies of the SAME type
# (t02's firmware update -> replacement pair, resolution -> resolution) never
# fire regardless. Run this ontology at both granularities and union the graphs.

CUSTOMER_SERVICE = Ontology(
    name="customer_service",
    description=(
        "Customer support threads: who reported what fault on which product and "
        "order, what caused it, what was promised, what actually got done, and "
        "which promise the company missed."
    ),
    entities=(
        EntityType("customer", "The person reporting the problem or asking for something: the account holder, the buyer. Not the support agent, however technical the customer sounds."),
        EntityType("agent", "A support representative, technician, or supervisor answering on the company's behalf. Named or self-introduced. Not the customer."),
        EntityType("product", "A branded thing the customer bought: a laptop, earphones, a router. Not a part inside one - use component."),
        EntityType("order", "A purchase, invoice, RMA or shipment identifier, in any spelling: 'A-99213', '#A-99213', 'order A99213', 'Invoice D-77401'. The transaction, not the product it bought."),
        EntityType("issue", "The fault or complaint as reported: the symptom, the wrong charge, the missing delivery. The problem, never the fix for it."),
        EntityType("resolution", "The remedy offered or carried out: a replacement, a repair, an RMA, a firmware update, a cancellation. The fix, never the problem. A remedy that was offered and then withdrawn is still a resolution - that is what `replaces` is for."),
        EntityType("refund", "Money going back to the customer: a refund, credit, goodwill payment, or pro-rata rebate, usually with an amount. A narrower thing than resolution - use it only when money moves."),
        EntityType("policy", "A standing company rule the agent invokes: a warranty term, an auto-renewal clause, a cooling-off period. Not a one-off promise the agent makes on this thread."),
        EntityType("channel", "The medium the contact came in on: email, chat, phone, callback, the portal. Not the shipping carrier and not the product."),
        EntityType("component", "A part or subsystem inside a product: the battery, the left earbud, the charging case, the power supply, the contact pins, firmware. Never the whole product."),
        EntityType("competitor", "A rival brand or retailer the customer names, usually as a threat to switch or a price comparison. Never one of the company's own products."),
        EntityType("sla", "A promised turnaround with a clock on it: 'two-business-day depot turnaround', '48-hour dispatch', 'a callback within 24 hours'. Only when a time commitment is actually stated - a general warranty is a policy, not an SLA."),
        EntityType("account_tier", "The service level on the account: Priority Care, Business Plan, standard cover. Not a product, and not the SLA that the tier happens to promise."),
    ),
    relations=(
        RelationType("reported_by", ("issue",), ("customer",),
                     "The customer who raised this problem. The reporter, not the agent who logged it.",
                     unique_head=True),
        RelationType("handled_by", ("issue",), ("agent",),
                     "The agent working the case. The one who takes it on, not anyone merely copied in."),
        RelationType("escalated_to", ("issue",), ("agent",),
                     "The case was handed up to this agent or supervisor because the first one could not settle it. Only on a real handover - answering a follow-up is not an escalation."),
        RelationType("concerns_product", ("issue", "resolution", "order"), ("product",),
                     "The product the head is about."),
        RelationType("about_order", ("issue", "resolution", "refund"), ("order",),
                     "The purchase the head refers to. Match order identifiers however they are written."),
        RelationType("owned_by", ("order", "product"), ("customer",),
                     "The customer who placed this order or owns this unit."),
        RelationType("caused_by", ("issue",), ("component", "product", "policy"),
                     "The root cause the agent actually identified. Only when a cause is stated - not merely the part that happens to be named in the same sentence."),
        RelationType("component_of", ("component",), ("product", "component"),
                     "The part belongs to this product, or to this larger part. Containment, not fault.",
                     unique_head=True, acyclic=True),
        RelationType("resolved_by", ("issue",), ("resolution", "refund"),
                     "The remedy that settled this issue."),
        RelationType("replaces", ("resolution", "refund"), ("resolution", "refund"),
                     "The head remedy supersedes the tail one: a refund agreed after a replacement was already offered, a replacement after a firmware fix failed. Head is the later, surviving remedy.",
                     acyclic=True, threshold=0.15),
        RelationType("compensated_with", ("customer", "issue"), ("refund",),
                     "Money or credit given to the head for the trouble."),
        RelationType("refers_to_policy", ("issue", "resolution", "agent"), ("policy",),
                     "The head is justified by, or blocked by, this stated rule."),
        RelationType("breaches", ("issue", "resolution"), ("sla",),
                     "The head missed the time commitment in the tail. Only for a commitment that was actually broken, not one merely quoted."),
        RelationType("contacted_via", ("customer",), ("channel",),
                     "The medium the customer got in touch on, as the customer states it: "
                     "'I tried the phone line and gave up, so chat it is', 'reply to this email'. "
                     "Headed by the customer rather than by the issue on purpose - see the note above."),
        RelationType("has_tier", ("customer",), ("account_tier",),
                     "The service level on this customer's account.",
                     unique_head=True),
        RelationType("mentions_competitor", ("customer", "issue"), ("competitor",),
                     "The customer named this rival brand or retailer, typically as a threat to leave or a price comparison."),
    ),
)
