"""Synthetic conversation memory stream for the knowledge-graph extraction demo.

Everything in this module is FABRICATED. Priya Raman, Northwind Logistics, and
every colleague, project, and artifact named here are invented for the purpose
of exercising an entity-extraction / coreference / entity-resolution pipeline.
No real person, company, or system is described.

The corpus models the assistant-side memory of a single user (Priya Raman) over
five chat sessions spanning 2026-02-10 through 2026-06-18. It is written to be
*hard* in specific, checkable ways:

  * Coreference stress. Facts are stated with bare pronouns ("he's the one
    who'll own the query layer") whose antecedent lives several turns earlier,
    and with definite descriptions ("the migration", "that repo", "the API").
  * Entity name variants. "Northwind Logistics" / "Northwind" / "NWL";
    "Order Graph Migration" / "the order graph project" / "order migration";
    "Tomás Ferreira" / "Tomas" / "Tomás".
  * Temporal invalidation. Two of Priya's stated preferences are contradicted by
    a later session (npm -> pnpm, Python -> Go). A correct pipeline should mark
    the earlier fact invalid-as-of the later session rather than storing both as
    simultaneously true. See ``CONTRADICTIONS``.

``GOLD_MEMORY_FACTS`` is the hand-written answer key: the triples a perfect
extractor + resolver should recover, expressed with canonical entity names.
"""

SESSIONS: list[dict] = [
    {
        "session_id": "s1",
        "timestamp": "2026-02-10T09:12:00Z",
        "title": "Order graph kickoff",
        "turns": [
            {
                "speaker": "user",
                "text": (
                    "morning. quick context since this is our first time working together — "
                    "I'm Priya Raman, I lead the Platform Engineering team at Northwind "
                    "Logistics. freight brokerage, mostly EU lanes, lots of very old order "
                    "data. I'm kicking off a project this week and want a sanity check "
                    "before I write it up."
                ),
            },
            {
                "speaker": "assistant",
                "text": (
                    "Morning Priya. Happy to be the sanity check. What's the project, and "
                    "what's the shape of the thing you're moving?"
                ),
            },
            {
                "speaker": "user",
                "text": (
                    "we're calling it the Order Graph Migration. the order service lives on "
                    "Postgres today — like nine years of accreted schema — and we want to "
                    "move the order/shipment/carrier relationships into Neo4j. the "
                    "read-heavy path is basically graph traversal that we've been faking "
                    "with six-way joins."
                ),
            },
            {
                "speaker": "assistant",
                "text": (
                    "That's a reasonable motivation — if the hot queries are variable-depth "
                    "traversals, joins will keep punishing you. Are you moving the whole "
                    "order service or just the read model first?"
                ),
            },
            {
                "speaker": "user",
                "text": (
                    "read model first, definitely. writes stay on Postgres until we trust "
                    "the thing. Dev Shah is doing the backend work on it — he's on my team, "
                    "he's been at Northwind about four years."
                ),
            },
            {
                "speaker": "assistant",
                "text": "Got it. Does Dev have graph query experience already, or is this new for him?",
            },
            {
                "speaker": "user",
                "text": (
                    "new. he's the one who'll own the Cypher query layer though, so he's "
                    "teaching himself as he goes. he's already rewritten the shipment "
                    "lookup twice and honestly the second one is decent."
                ),
            },
            {
                "speaker": "assistant",
                "text": (
                    "Good sign. Who else is on the team, and is anyone covering the "
                    "operational side — backups, cluster sizing, that sort of thing?"
                ),
            },
            {
                "speaker": "user",
                "text": (
                    "Tomás Ferreira. he's our SRE, reports into me along with Dev. he'll do "
                    "the cluster and the alerting once we're past the prototype. right now "
                    "he's buried in the Fleet Telemetry Pipeline, which is its own saga."
                ),
            },
            {
                "speaker": "assistant",
                "text": "Understood. Anything about tooling or workflow I should hold onto for next time?",
            },
            {
                "speaker": "user",
                "text": (
                    "yeah a couple. keep all the JS tooling on npm — I know pnpm is what "
                    "everyone's excited about but npm is what our CI knows and I don't want "
                    "to relearn a lockfile format this quarter. so npm, please, in anything "
                    "you scaffold for me."
                ),
            },
            {
                "speaker": "assistant",
                "text": "Noted — npm for package management. Anything on the CI side itself?",
            },
            {
                "speaker": "user",
                "text": (
                    "one hard no: no Jenkins. I inherited a Jenkins estate at my last job "
                    "and I'm not touching Jenkins again, ever. if a suggestion starts with "
                    "'a small groovy script' I will close the tab."
                ),
            },
            {
                "speaker": "assistant",
                "text": (
                    "Fair. npm yes, Jenkins never. Anything else before you write up the "
                    "kickoff?"
                ),
            },
            {
                "speaker": "user",
                "text": (
                    "scheduling, since you'll help me draft invites eventually — I don't do "
                    "meetings before 09:00 CET. I'm in Berlin now, moved from Bangalore in "
                    "January, and the early block is the only quiet time I get before the "
                    "day eats me."
                ),
            },
            {
                "speaker": "assistant",
                "text": (
                    "Understood — Berlin, nothing scheduled before 09:00 CET. I'll keep all "
                    "of that in mind. Good luck with the kickoff."
                ),
            },
        ],
    },
    {
        "session_id": "s2",
        "timestamp": "2026-03-04T10:05:00Z",
        "title": "Migration tooling and telemetry detour",
        "turns": [
            {
                "speaker": "user",
                "text": (
                    "back. so the order graph project is real now, we have a prototype "
                    "loading about 2M orders. I need to decide what to write the migration "
                    "tooling in — the ETL that pulls from Postgres, reshapes, and bulk loads "
                    "into Neo4j."
                ),
            },
            {
                "speaker": "assistant",
                "text": (
                    "What are the candidate languages, and who's going to be maintaining the "
                    "tooling after the cutover?"
                ),
            },
            {
                "speaker": "user",
                "text": (
                    "I want Python for it. the transform logic is gnarly and pandas plus the "
                    "neo4j driver gets us there fastest, and half the team can already read "
                    "Python. so: Python for the migration tooling, that's the call."
                ),
            },
            {
                "speaker": "assistant",
                "text": (
                    "Reasonable. Python it is for the ETL. Do you want the loader batching "
                    "through the driver, or generating CSV for the bulk importer?"
                ),
            },
            {
                "speaker": "user",
                "text": (
                    "batching for now. one constraint you need to bake into anything you "
                    "help me design: no customer PII in dev environments. we're mid SOC 2 "
                    "and the auditors were very clear. so the ETL has to have a scrub step "
                    "before anything lands anywhere that isn't prod."
                ),
            },
            {
                "speaker": "assistant",
                "text": (
                    "Understood — SOC 2, no customer PII outside prod. That probably means a "
                    "deterministic pseudonymizer so dev data still joins correctly."
                ),
            },
            {
                "speaker": "user",
                "text": (
                    "exactly what I was going to ask for. write that down as a hard "
                    "constraint, it's not a preference, it's an audit finding waiting to "
                    "happen."
                ),
            },
            {
                "speaker": "assistant",
                "text": "Recorded as a hard constraint. Anything blocking the ETL work right now?",
            },
            {
                "speaker": "user",
                "text": (
                    "not the ETL, no. but Tomás is underwater. the Fleet Telemetry Pipeline "
                    "is ingesting from about 1,400 trucks and the backpressure handling is "
                    "wrong, so he's been firefighting for two weeks."
                ),
            },
            {
                "speaker": "assistant",
                "text": "Is that pipeline going to compete with the migration for his time?",
            },
            {
                "speaker": "user",
                "text": (
                    "a bit. he's frustrated with the whole Kafka consumer setup on it, keeps "
                    "saying we picked the wrong retention policy, and he's probably right. "
                    "but it's NWL's flagship customer-facing feature so it wins."
                ),
            },
            {
                "speaker": "assistant",
                "text": (
                    "Makes sense. Do you want me to draft a short design doc for the ETL so "
                    "the team has something to review?"
                ),
            },
            {
                "speaker": "user",
                "text": (
                    "yes but not in Confluence. no more Confluence for design docs — nobody "
                    "reads it, search is a joke, and we've got four competing spaces. "
                    "markdown in the repo, reviewed as a PR, that's it."
                ),
            },
            {
                "speaker": "assistant",
                "text": "Markdown-in-repo it is. Which repo should it live in?",
            },
            {
                "speaker": "user",
                "text": (
                    "that repo Dev set up, platform/order-graph. he's got the CI wired "
                    "already so a docs PR will just run lint and be done. and it's npm, per "
                    "usual."
                ),
            },
            {
                "speaker": "assistant",
                "text": (
                    "Got it — I'll draft the ETL design doc as markdown in platform/"
                    "order-graph, Python implementation, PII scrub in the pipeline."
                ),
            },
        ],
    },
    {
        "session_id": "s3",
        "timestamp": "2026-04-15T09:30:00Z",
        "title": "Q2 review prep and the two blockers",
        "turns": [
            {
                "speaker": "user",
                "text": (
                    "I have the Q2 Platform Review on thursday and I need to be honest about "
                    "why the migration has slipped. can you help me structure it? I don't "
                    "want it to read like excuses."
                ),
            },
            {
                "speaker": "assistant",
                "text": (
                    "Sure. Start with what's actually blocking rather than what's late — "
                    "what's holding the Order Graph Migration up?"
                ),
            },
            {
                "speaker": "user",
                "text": (
                    "two things and they're both real. first, we can't cut over the read "
                    "path until Carrier Rate API v2 ships, because the graph model pulls "
                    "rate cards from it and v1 doesn't expose the effective-date fields. so "
                    "the order migration is blocked by that API, full stop."
                ),
            },
            {
                "speaker": "assistant",
                "text": "Who owns the Carrier Rate API v2?",
            },
            {
                "speaker": "user",
                "text": (
                    "Dev, same as the query layer, which is part of the problem — I've got "
                    "one person on two critical paths. he's splitting roughly 60/40 and "
                    "neither half is happy about it."
                ),
            },
            {
                "speaker": "assistant",
                "text": "And the second blocker?",
            },
            {
                "speaker": "user",
                "text": (
                    "the Neo4j enterprise license. it needs sign-off from Marcus Webb — he's "
                    "our CFO — and it's been sitting in his queue since february. we're on "
                    "community edition in dev which means no RBAC, which means I can't even "
                    "pretend we're SOC 2 compliant on that cluster."
                ),
            },
            {
                "speaker": "assistant",
                "text": (
                    "So the license approval isn't a nice-to-have, it's on the compliance "
                    "path. Has he given a reason for the delay?"
                ),
            },
            {
                "speaker": "user",
                "text": (
                    "he wants a three-year TCO comparison against staying on Postgres. which "
                    "is fair, honestly, but nobody told me that was the gate until march."
                ),
            },
            {
                "speaker": "assistant",
                "text": (
                    "That's a clean story for the review then: one dependency blocker, one "
                    "approval blocker, both with named owners. Do you have anything written "
                    "down that captures the decision itself?"
                ),
            },
            {
                "speaker": "user",
                "text": (
                    "yeah — I wrote Architecture Decision Record 014 last month, 'Order "
                    "read model on Neo4j'. it covers the model, the rejected alternatives, "
                    "and the rollback plan. ADR 014 is basically the artifact I'd point "
                    "Marcus at for the TCO argument."
                ),
            },
            {
                "speaker": "assistant",
                "text": "Did anyone outside the team review it?",
            },
            {
                "speaker": "user",
                "text": (
                    "Elena Vasquez did. she's the consultant we brought in on graph modeling "
                    "— external, works with us maybe two days a month. she suggested we "
                    "model the carrier rate as a relationship property with an interval "
                    "rather than a separate node, and that cut our node count by like 40%."
                ),
            },
            {
                "speaker": "assistant",
                "text": "That's worth putting in the review — it's a concrete design win, not just status.",
            },
            {
                "speaker": "user",
                "text": (
                    "agreed. also Tomas finally got the telemetry backpressure fixed, so the "
                    "fleet pipeline is stable and he's got maybe 30% of his week back for "
                    "the graph cluster work."
                ),
            },
            {
                "speaker": "assistant",
                "text": (
                    "Good. So: Tomás freed up, Elena's model change landed, blocked on the "
                    "Carrier Rate API and the license. I'll draft that into three slides."
                ),
            },
        ],
    },
    {
        "session_id": "s4",
        "timestamp": "2026-05-20T09:45:00Z",
        "title": "Tooling switch and the workshop",
        "turns": [
            {
                "speaker": "user",
                "text": (
                    "small housekeeping thing first: stop using npm in anything you generate "
                    "for me. we've switched to pnpm across the platform repos."
                ),
            },
            {
                "speaker": "assistant",
                "text": (
                    "Noted — pnpm from here on. That's a reversal from what you told me in "
                    "February; what changed?"
                ),
            },
            {
                "speaker": "user",
                "text": (
                    "the monorepo happened, basically. we went from three repos to one "
                    "workspace and npm's install times were like four minutes on CI. pnpm "
                    "took it to 50 seconds. I was wrong, I prefer pnpm now, please use it "
                    "everywhere going forward."
                ),
            },
            {
                "speaker": "assistant",
                "text": (
                    "Understood — pnpm supersedes npm as your preference as of today. Still "
                    "no Jenkins, I assume?"
                ),
            },
            {
                "speaker": "user",
                "text": "correct. that one is permanent. take it to my grave.",
            },
            {
                "speaker": "assistant",
                "text": "Fair enough. How's the migration going since the Q2 review?",
            },
            {
                "speaker": "user",
                "text": (
                    "better. Elena ran a Neo4j training workshop for us two weeks ago, full "
                    "day, and it unstuck a lot. Dev went, Tomás went, I sat in for the "
                    "modeling half. Dev's Cypher is genuinely good now — he's writing "
                    "query plans I wouldn't have thought of."
                ),
            },
            {
                "speaker": "assistant",
                "text": "That's a real change from February. Did the workshop cover operational stuff too?",
            },
            {
                "speaker": "user",
                "text": (
                    "some. clustering, backup strategy. Tomás got what he needed out of it. "
                    "he's now sketching the prod topology for the graph cluster, assuming we "
                    "ever get to have one."
                ),
            },
            {
                "speaker": "assistant",
                "text": "That 'assuming' is doing some work. Still waiting on the license?",
            },
            {
                "speaker": "user",
                "text": (
                    "yep. he still hasn't approved it. I sent the TCO breakdown three weeks "
                    "ago, got a 'thanks, reviewing' and nothing since. it's the single "
                    "biggest risk to the whole order graph project right now."
                ),
            },
            {
                "speaker": "assistant",
                "text": "Is there a date after which the timeline breaks?",
            },
            {
                "speaker": "user",
                "text": (
                    "yes and this is the thing I need you to remember. Northwind has a Black "
                    "Friday change freeze — November 15 through December 2, absolutely "
                    "nothing ships to prod, it's peak freight season. so if we haven't cut "
                    "over by early november the migration slides to january."
                ),
            },
            {
                "speaker": "assistant",
                "text": (
                    "Recorded: company-wide change freeze Nov 15 – Dec 2. That makes early "
                    "November your real deadline, not end of year."
                ),
            },
            {
                "speaker": "user",
                "text": (
                    "right. and Carrier Rate API v2 is in staging now, so that blocker is "
                    "close to clearing at least. one down, one to go, hopefully."
                ),
            },
            {
                "speaker": "assistant",
                "text": (
                    "Good. Summary I'm holding: pnpm now, no Jenkins, freeze Nov 15 – Dec 2, "
                    "license still pending with the CFO."
                ),
            },
        ],
    },
    {
        "session_id": "s5",
        "timestamp": "2026-06-18T09:20:00Z",
        "title": "Standardizing on Go",
        "turns": [
            {
                "speaker": "user",
                "text": (
                    "another reversal for you, sorry. the migration tooling isn't Python "
                    "anymore. the team standardized on Go for it and we've already ported "
                    "the loader."
                ),
            },
            {
                "speaker": "assistant",
                "text": (
                    "Understood — Go for the migration tooling, replacing the Python "
                    "decision from March. What drove it?"
                ),
            },
            {
                "speaker": "user",
                "text": (
                    "two things. the Python loader was doing 2M orders in about 40 minutes "
                    "and we need the cutover window under 15. and the rest of NWL's backend "
                    "is Go, so on-call people could actually read it. so yeah — Go for the "
                    "migration tooling, that's the standard now, Python's out for this."
                ),
            },
            {
                "speaker": "assistant",
                "text": (
                    "Noted, and the earlier Python preference is superseded. Did the "
                    "PII-scrub step survive the port?"
                ),
            },
            {
                "speaker": "user",
                "text": (
                    "yes, first thing we ported. that constraint hasn't moved — no customer "
                    "PII in dev, SOC 2 is still very much a live thing."
                ),
            },
            {
                "speaker": "assistant",
                "text": "Good. Where does that leave the blockers we talked about in May?",
            },
            {
                "speaker": "user",
                "text": (
                    "Carrier Rate API v2 shipped on the 3rd. Dev pushed it out and it's been "
                    "stable, so the order migration isn't blocked on the API anymore."
                ),
            },
            {
                "speaker": "assistant",
                "text": "And the license?",
            },
            {
                "speaker": "user",
                "text": (
                    "signed. finally. Marcus approved it two weeks ago after I walked him "
                    "through ADR 014 line by line. so we're unblocked on both, which is "
                    "terrifying because now it's just execution."
                ),
            },
            {
                "speaker": "assistant",
                "text": "That's a good problem. Who's doing what for the cutover?",
            },
            {
                "speaker": "user",
                "text": (
                    "Dev owns the query layer and the dual-read shim. Tomás owns the cluster "
                    "and the runbook. I own the go/no-go and the conversation with the "
                    "carrier ops team, who are going to hate this no matter how well it "
                    "goes."
                ),
            },
            {
                "speaker": "assistant",
                "text": "Is the November freeze still the binding constraint?",
            },
            {
                "speaker": "user",
                "text": (
                    "still binding, Nov 15 to Dec 2, and I'm targeting a cutover the last "
                    "week of october to leave slack. that's the goal I'm running the whole "
                    "team against — order service off Postgres before the freeze."
                ),
            },
            {
                "speaker": "assistant",
                "text": "Want me to draft the runbook skeleton in the repo, Go-flavored, pnpm for the JS bits?",
            },
            {
                "speaker": "user",
                "text": (
                    "yes. markdown in platform/order-graph like last time, definitley not "
                    "Confluence. and schedule nothing before 09:00 for the go/no-go, I want "
                    "to be awake for that one."
                ),
            },
            {
                "speaker": "assistant",
                "text": (
                    "Understood. Runbook in platform/order-graph, Go tooling, pnpm, no "
                    "meetings before 09:00 CET."
                ),
            },
        ],
    },
]


GOLD_MEMORY_FACTS: list[tuple[str, str, str]] = [
    # Identity and org
    ("Priya Raman", "works_at", "Northwind Logistics"),
    ("Priya Raman", "part_of", "Platform Engineering"),
    ("Platform Engineering", "part_of", "Northwind Logistics"),
    ("Priya Raman", "located_in", "Berlin"),
    ("Dev Shah", "works_at", "Northwind Logistics"),
    ("Dev Shah", "part_of", "Platform Engineering"),
    ("Tomás Ferreira", "part_of", "Platform Engineering"),
    ("Marcus Webb", "works_at", "Northwind Logistics"),
    ("Priya Raman", "collaborates_with", "Dev Shah"),
    ("Priya Raman", "collaborates_with", "Tomás Ferreira"),
    ("Priya Raman", "collaborates_with", "Marcus Webb"),
    ("Priya Raman", "collaborates_with", "Elena Vasquez"),
    # Projects and assignments
    ("Priya Raman", "works_on", "Order Graph Migration"),
    ("Dev Shah", "assigned_to", "Order Graph Migration"),
    ("Dev Shah", "works_on", "Carrier Rate API v2"),
    ("Tomás Ferreira", "works_on", "Fleet Telemetry Pipeline"),
    ("Elena Vasquez", "works_on", "Order Graph Migration"),
    ("Order Graph Migration", "uses_tool", "Neo4j"),
    # Blockers
    ("Order Graph Migration", "blocked_by", "Carrier Rate API v2"),
    ("Order Graph Migration", "blocked_by", "Neo4j license approval"),
    ("Neo4j license approval", "assigned_to", "Marcus Webb"),
    # Preferences (the npm and Python entries are superseded; see CONTRADICTIONS)
    ("Priya Raman", "prefers", "npm"),
    ("Priya Raman", "prefers", "pnpm"),
    ("Priya Raman", "prefers", "Python"),
    ("Priya Raman", "prefers", "Go"),
    ("Priya Raman", "avoids", "Jenkins"),
    ("Priya Raman", "avoids", "Confluence"),
    # Constraints
    ("Priya Raman", "has_constraint", "No meetings before 09:00 CET"),
    ("Priya Raman", "has_constraint", "No customer PII in dev environments"),
    ("Northwind Logistics", "has_constraint", "Black Friday change freeze"),
    # Skills, artifacts, events, goals
    ("Dev Shah", "has_skill", "Cypher"),
    ("Priya Raman", "produced", "Architecture Decision Record 014"),
    ("Priya Raman", "attended", "Q2 Platform Review"),
    ("Dev Shah", "attended", "Neo4j training workshop"),
    ("Tomás Ferreira", "attended", "Neo4j training workshop"),
    ("Priya Raman", "pursues", "Cut the order service over off Postgres before the change freeze"),
]


CONTRADICTIONS: list[dict] = [
    {
        "subject": "Priya Raman",
        "relation": "prefers",
        "old": "npm",
        "new": "pnpm",
        "old_session": "s1",
        "new_session": "s4",
    },
    {
        "subject": "Priya Raman",
        "relation": "prefers",
        "old": "Python",
        "new": "Go",
        "old_session": "s2",
        "new_session": "s5",
    },
]
