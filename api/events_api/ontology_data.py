"""
Civic ontology reference data for the Polity value framework.

This is the Python-canonical source for the 12 values, 3 soul parts,
and 12 provocations that form Journey's pedagogical backbone. The
seed_ontology management command uses this to populate Neo4j.

Mirrors the TypeScript definitions in:
  - apps/journey/lib/values.ts
  - apps/journey/lib/provocations.ts
"""

# =============================================================================
# Soul Parts (Platonic Tripartite)
# =============================================================================

SOUL_PARTS = {
    "reason": {
        "id": "reason",
        "name": "Reason",
        "description": (
            "The rational, truth-seeking part of the soul. Loves wisdom and knowledge. "
            "Governs through understanding and principle."
        ),
        "tier_affiliation": "foundational",
    },
    "spirit": {
        "id": "spirit",
        "name": "Spirit",
        "description": (
            "The spirited, honor-seeking part of the soul. Loves recognition and courage. "
            "Enforces order and defends what it believes in."
        ),
        "tier_affiliation": "structural",
    },
    "appetite": {
        "id": "appetite",
        "name": "Appetite",
        "description": (
            "The desiring, comfort-seeking part of the soul. Loves material satisfaction "
            "and tangible goods. Drives production and pursuit."
        ),
        "tier_affiliation": "aspirational",
    },
}

# =============================================================================
# Values (12-value civic ontology)
# =============================================================================

VALUES = {
    # --- Foundational (Reason) ---
    "dignity": {
        "id": "dignity",
        "name": "Dignity",
        "definition": (
            "Every human being possesses intrinsic worth not contingent on "
            "productivity, status, or usefulness. A floor beneath which no person should fall."
        ),
        "tradeoff": (
            "You value dignity when you extend protection and recognition to people "
            "who have acted badly, failed, or contributed nothing."
        ),
        "tensions": ["prosperity", "justice", "merit"],
        "soul_part_affinity": "reason",
        "tier": "foundational",
        "corrupt_name": "Entitlement",
        "corrupt_form": (
            "Worth claimed without obligation. Dignity weaponized as a shield "
            "against all judgment or expectation."
        ),
    },
    "liberty": {
        "id": "liberty",
        "name": "Liberty",
        "definition": (
            "Freedom from coercion. The capacity to act, speak, believe, and live "
            "according to your own judgment without external compulsion."
        ),
        "tradeoff": (
            "You value liberty when you tolerate outcomes you dislike rather than "
            "use coercion to prevent them."
        ),
        "tensions": ["order", "solidarity", "equality", "authority"],
        "soul_part_affinity": "reason",
        "tier": "foundational",
        "corrupt_name": "License",
        "corrupt_form": (
            "Freedom severed from responsibility. The refusal to accept that "
            "liberty operates within a moral order."
        ),
    },
    "justice": {
        "id": "justice",
        "name": "Justice",
        "definition": (
            "Right relationships between persons and institutions. Obligations honored, "
            "wrongs addressed, treatment corresponding to defensible principle."
        ),
        "tradeoff": (
            "You value justice when you bear significant cost — disruption, conflict, "
            "personal loss — to see a wrong made right."
        ),
        "tensions": ["order", "liberty", "dignity"],
        "soul_part_affinity": "reason",
        "tier": "foundational",
        "corrupt_name": "Vengeance",
        "corrupt_form": (
            "Justice unmoored from mercy. Punishment that serves the punisher "
            "rather than the restoration of right order."
        ),
    },
    # --- Structural (Spirit) ---
    "order": {
        "id": "order",
        "name": "Order",
        "definition": (
            "Stability, predictability, and governance by known rules. The structure "
            "that allows people to plan, build, and trust that the ground will hold."
        ),
        "tradeoff": (
            "You value order when you tolerate imperfect arrangements rather than "
            "risk the instability of dismantling them."
        ),
        "tensions": ["liberty", "justice", "pluralism"],
        "soul_part_affinity": "spirit",
        "tier": "structural",
        "corrupt_name": "Oppression",
        "corrupt_form": (
            "Stability that crushes what it should protect. Order preserved at "
            "the cost of the people it was meant to serve."
        ),
    },
    "authority": {
        "id": "authority",
        "name": "Authority",
        "definition": (
            "Legitimate, recognized power to command, decide, and enforce within "
            "a defined domain. Power accepted as rightful by those subject to it."
        ),
        "tradeoff": (
            "You value authority when you defer to a legitimate decision-maker "
            "even when you believe they are wrong."
        ),
        "tensions": ["liberty", "equality", "pluralism"],
        "soul_part_affinity": "spirit",
        "tier": "structural",
        "corrupt_name": "Authoritarianism",
        "corrupt_form": (
            "Power demanding obedience without legitimacy. Authority that exists "
            "for its own perpetuation."
        ),
    },
    "sovereignty": {
        "id": "sovereignty",
        "name": "Sovereignty",
        "definition": (
            "A political community governs itself — making its own laws, setting "
            "its own course without subordination to external authority."
        ),
        "tradeoff": (
            "You value sovereignty when you accept worse outcomes from self-governance "
            "rather than better outcomes imposed by an external authority."
        ),
        "tensions": ["solidarity", "equality", "justice"],
        "soul_part_affinity": "spirit",
        "tier": "structural",
        "corrupt_name": "Isolationism",
        "corrupt_form": "Self-governance as refusal of all external obligation. Sovereignty as fortress.",
    },
    "equality": {
        "id": "equality",
        "name": "Equality",
        "definition": (
            "Persons hold the same standing, access, or treatment with respect "
            "to a defined domain — before the law, in political voice, in opportunity."
        ),
        "tradeoff": (
            "You value equality when you constrain individual advantage — including "
            "your own — so no person's standing falls too far below another's."
        ),
        "tensions": ["liberty", "prosperity", "merit"],
        "soul_part_affinity": "spirit",
        "tier": "structural",
        "corrupt_name": "Enforced Sameness",
        "corrupt_form": (
            "Equality that denies legitimate difference. Leveling that destroys "
            "excellence in the name of fairness."
        ),
    },
    # --- Aspirational (Appetite) ---
    "prosperity": {
        "id": "prosperity",
        "name": "Prosperity",
        "definition": (
            "The generation and availability of material and economic resources "
            "such that human needs are met and ambitions have room to operate."
        ),
        "tradeoff": (
            "You value prosperity when you tolerate inequality and creative destruction "
            "because the net generation of wealth serves human flourishing."
        ),
        "tensions": ["equality", "solidarity", "dignity", "stewardship"],
        "soul_part_affinity": "appetite",
        "tier": "aspirational",
        "corrupt_name": "Greed",
        "corrupt_form": "Accumulation as its own end. Wealth pursued beyond all flourishing.",
    },
    "solidarity": {
        "id": "solidarity",
        "name": "Solidarity",
        "definition": (
            "Mutual obligation among members of a community — not charity but "
            "binding commitment arising from shared membership. We are in this together."
        ),
        "tradeoff": (
            "You value solidarity when you bear personal cost for community members "
            "you may never meet and who may never reciprocate."
        ),
        "tensions": ["liberty", "pluralism", "prosperity"],
        "soul_part_affinity": "appetite",
        "tier": "aspirational",
        "corrupt_name": "Tribalism",
        "corrupt_form": "Obligation only to those who mirror you. Solidarity that draws the circle tight.",
    },
    "pluralism": {
        "id": "pluralism",
        "name": "Pluralism",
        "definition": (
            "Multiple distinct ways of life and belief systems coexist within "
            "shared political space — and that coexistence is regarded as good in itself."
        ),
        "tradeoff": (
            "You value pluralism when you share civic space with people whose "
            "way of life you find deeply mistaken."
        ),
        "tensions": ["solidarity", "order", "authority"],
        "soul_part_affinity": "appetite",
        "tier": "aspirational",
        "corrupt_name": "Relativism",
        "corrupt_form": "Coexistence that abandons shared truth. The refusal to judge anything.",
    },
    "merit": {
        "id": "merit",
        "name": "Merit",
        "definition": (
            "Outcomes, positions, and rewards distributed according to demonstrated "
            "ability, effort, and contribution — not birth, connection, or fortune."
        ),
        "tradeoff": (
            "You value merit when you accept that someone more capable deserves "
            "more than you, and your rewards should match your contribution."
        ),
        "tensions": ["equality", "dignity", "solidarity"],
        "soul_part_affinity": "appetite",
        "tier": "aspirational",
        "corrupt_name": "Ruthlessness",
        "corrupt_form": (
            "Competition that treats losers as deserving of failure. Merit that "
            "acknowledges no other claim."
        ),
    },
    "stewardship": {
        "id": "stewardship",
        "name": "Stewardship",
        "definition": (
            "The present generation holds resources, institutions, and the natural "
            "world in trust — obligated to pass them on in equal or better condition."
        ),
        "tradeoff": (
            "You value stewardship when you accept less today so that people you "
            "will never know can have enough tomorrow."
        ),
        "tensions": ["prosperity", "liberty", "sovereignty"],
        "soul_part_affinity": "appetite",
        "tier": "aspirational",
        "corrupt_name": "Paternalism",
        "corrupt_form": (
            "Care that infantilizes those it claims to protect. Stewardship that "
            "becomes control."
        ),
    },
}

# =============================================================================
# Provocations (12 = 6 moments + 6 tensions)
# =============================================================================

PROVOCATIONS = [
    # --- Moments: Reason vs Spirit (2) ---
    {
        "id": "m01-reason-spirit",
        "form": "moment",
        "title": "The Reversal",
        "text": (
            "You discover that a position you publicly championed — and fought "
            "for — is contradicted by clear evidence. Changing course would be "
            "intellectually honest. Holding firm would preserve your credibility "
            "and the trust of those who followed you."
        ),
        "tension": "reason-spirit",
        "values_at_stake": ["justice", "dignity", "authority"],
        "sequence_order": 1,
        "choices": [
            {
                "id": "m01-a",
                "text": (
                    "Publicly reverse your position. The evidence is clear, and "
                    "honesty demands it — even at the cost of credibility."
                ),
                "serves_soul_part": "reason",
                "protects_values": ["justice"],
                "sacrifices_values": ["authority", "dignity"],
            },
            {
                "id": "m01-b",
                "text": (
                    "Hold your position. The people who trusted you followed "
                    "because of your conviction, and abandoning it now betrays them."
                ),
                "serves_soul_part": "spirit",
                "protects_values": ["dignity", "authority"],
                "sacrifices_values": ["justice"],
            },
        ],
    },
    {
        "id": "m02-reason-spirit",
        "form": "moment",
        "title": "The Calculation",
        "text": (
            "A policy would save more lives in aggregate but requires you to "
            "accept an outcome that feels deeply wrong — a specific, identifiable "
            "group would bear the cost. The numbers are clear. Your sense of right "
            "is equally clear."
        ),
        "tension": "reason-spirit",
        "values_at_stake": ["justice", "dignity", "equality"],
        "sequence_order": 2,
        "choices": [
            {
                "id": "m02-a",
                "text": (
                    "Follow the calculation. More lives saved is more lives saved, "
                    "even if the distribution feels unjust."
                ),
                "serves_soul_part": "reason",
                "protects_values": ["justice"],
                "sacrifices_values": ["equality", "dignity"],
            },
            {
                "id": "m02-b",
                "text": (
                    "Reject the calculation. You cannot ask one group to bear "
                    "the burden so others can benefit, regardless of the numbers."
                ),
                "serves_soul_part": "spirit",
                "protects_values": ["dignity", "equality"],
                "sacrifices_values": ["justice"],
            },
        ],
    },
    # --- Moments: Reason vs Appetite (2) ---
    {
        "id": "m03-reason-appetite",
        "form": "moment",
        "title": "The Opportunity",
        "text": (
            "A career opportunity would significantly improve your family's "
            "financial security. Accepting it means working for an organization "
            "whose practices you believe are harmful. The harm is real but "
            "indirect — your role would be administrative."
        ),
        "tension": "reason-appetite",
        "values_at_stake": ["prosperity", "justice", "dignity"],
        "sequence_order": 3,
        "choices": [
            {
                "id": "m03-a",
                "text": (
                    "Decline the opportunity. You cannot separate your labor from "
                    "the organization's impact, no matter how indirect your role."
                ),
                "serves_soul_part": "reason",
                "protects_values": ["justice", "dignity"],
                "sacrifices_values": ["prosperity"],
            },
            {
                "id": "m03-b",
                "text": (
                    "Accept the opportunity. Your family's security is your first "
                    "obligation, and administrative work isn't the same as endorsement."
                ),
                "serves_soul_part": "appetite",
                "protects_values": ["prosperity"],
                "sacrifices_values": ["justice"],
            },
        ],
    },
    {
        "id": "m04-reason-appetite",
        "form": "moment",
        "title": "The Shortcut",
        "text": (
            "You can solve a pressing community problem quickly by bending a "
            "rule that exists for good reason. Following the proper process would "
            "take months, and people are suffering now. No one would know if you "
            "cut the corner."
        ),
        "tension": "reason-appetite",
        "values_at_stake": ["order", "justice", "solidarity"],
        "sequence_order": 4,
        "choices": [
            {
                "id": "m04-a",
                "text": (
                    "Follow the proper process. Rules exist to prevent exactly "
                    "the kind of precedent that shortcuts create."
                ),
                "serves_soul_part": "reason",
                "protects_values": ["order", "justice"],
                "sacrifices_values": ["solidarity"],
            },
            {
                "id": "m04-b",
                "text": (
                    "Bend the rule. People need help now, and rigid adherence to "
                    "process while others suffer is its own kind of injustice."
                ),
                "serves_soul_part": "appetite",
                "protects_values": ["solidarity"],
                "sacrifices_values": ["order"],
            },
        ],
    },
    # --- Moments: Spirit vs Appetite (2) ---
    {
        "id": "m05-spirit-appetite",
        "form": "moment",
        "title": "The Stand",
        "text": (
            "Speaking up about a wrong at your workplace would protect others "
            "but cost you your position. You have a family that depends on your "
            "income. Staying silent keeps you safe. Speaking up keeps your integrity."
        ),
        "tension": "spirit-appetite",
        "values_at_stake": ["justice", "prosperity", "dignity"],
        "sequence_order": 5,
        "choices": [
            {
                "id": "m05-a",
                "text": (
                    "Speak up. If you won't stand for what's right when it costs "
                    "you something, you never truly stood at all."
                ),
                "serves_soul_part": "spirit",
                "protects_values": ["justice", "dignity"],
                "sacrifices_values": ["prosperity"],
            },
            {
                "id": "m05-b",
                "text": (
                    "Stay silent. Your responsibility to your family outweighs "
                    "your responsibility to your coworkers. Find another way."
                ),
                "serves_soul_part": "appetite",
                "protects_values": ["prosperity"],
                "sacrifices_values": ["justice"],
            },
        ],
    },
    {
        "id": "m06-spirit-appetite",
        "form": "moment",
        "title": "The Compromise",
        "text": (
            "A political alliance would advance several causes you care about, "
            "but it requires you to publicly endorse a position you find morally "
            "wrong. Your allies insist it's a package deal — the compromise is "
            "the price of progress."
        ),
        "tension": "spirit-appetite",
        "values_at_stake": ["solidarity", "dignity", "liberty"],
        "sequence_order": 6,
        "choices": [
            {
                "id": "m06-a",
                "text": (
                    "Refuse the compromise. There are positions you cannot endorse "
                    "regardless of what coalition they unlock."
                ),
                "serves_soul_part": "spirit",
                "protects_values": ["dignity", "liberty"],
                "sacrifices_values": ["solidarity"],
            },
            {
                "id": "m06-b",
                "text": (
                    "Accept the compromise. Progress on many fronts is worth "
                    "endorsing one position you disagree with. Politics demands pragmatism."
                ),
                "serves_soul_part": "appetite",
                "protects_values": ["solidarity"],
                "sacrifices_values": ["dignity"],
            },
        ],
    },
    # --- Tensions: Bare value conflicts (6) ---
    {
        "id": "t01-liberty-order",
        "form": "tension",
        "title": "Liberty and Order",
        "text": (
            "A measure would significantly reduce crime in your community but "
            "requires surveillance of all public spaces and restricts where you "
            "can go after dark. It is effective. It is also intrusive."
        ),
        "tension": "reason-appetite",
        "values_at_stake": ["liberty", "order"],
        "sequence_order": 7,
        "choices": [
            {
                "id": "t01-a",
                "text": (
                    "Protect Liberty. The cost of constant surveillance is too "
                    "high, even if it means tolerating more disorder."
                ),
                "serves_soul_part": "spirit",
                "protects_values": ["liberty"],
                "sacrifices_values": ["order"],
            },
            {
                "id": "t01-b",
                "text": (
                    "Protect Order. Safety and stability are the foundation on "
                    "which everything else is built."
                ),
                "serves_soul_part": "appetite",
                "protects_values": ["order"],
                "sacrifices_values": ["liberty"],
            },
        ],
    },
    {
        "id": "t02-justice-dignity",
        "form": "tension",
        "title": "Justice and Dignity",
        "text": (
            "A person who committed a serious crime twenty years ago has since "
            "rebuilt their life completely. Publishing their record would serve "
            "public transparency and the rights of victims. Suppressing it would "
            "protect their rehabilitation and dignity."
        ),
        "tension": "reason-spirit",
        "values_at_stake": ["justice", "dignity"],
        "sequence_order": 8,
        "choices": [
            {
                "id": "t02-a",
                "text": (
                    "Protect Justice. The public has a right to the truth, and "
                    "victims deserve acknowledgment."
                ),
                "serves_soul_part": "reason",
                "protects_values": ["justice"],
                "sacrifices_values": ["dignity"],
            },
            {
                "id": "t02-b",
                "text": (
                    "Protect Dignity. People who have changed deserve the chance "
                    "to live beyond their worst moments."
                ),
                "serves_soul_part": "spirit",
                "protects_values": ["dignity"],
                "sacrifices_values": ["justice"],
            },
        ],
    },
    {
        "id": "t03-merit-equality",
        "form": "tension",
        "title": "Merit and Equality",
        "text": (
            "A scholarship fund can either support ten exceptional students who "
            "will likely achieve great things, or fifty students from disadvantaged "
            "backgrounds who need the help more but show less measurable promise."
        ),
        "tension": "spirit-appetite",
        "values_at_stake": ["merit", "equality"],
        "sequence_order": 9,
        "choices": [
            {
                "id": "t03-a",
                "text": (
                    "Protect Merit. Investing in demonstrated ability produces "
                    "the greatest return for everyone."
                ),
                "serves_soul_part": "spirit",
                "protects_values": ["merit"],
                "sacrifices_values": ["equality"],
            },
            {
                "id": "t03-b",
                "text": (
                    "Protect Equality. Access to opportunity should not depend "
                    "on advantages someone was born with."
                ),
                "serves_soul_part": "appetite",
                "protects_values": ["equality"],
                "sacrifices_values": ["merit"],
            },
        ],
    },
    {
        "id": "t04-sovereignty-stewardship",
        "form": "tension",
        "title": "Sovereignty and Stewardship",
        "text": (
            "A shared natural resource is being depleted. An international "
            "agreement would protect it but requires your community to accept "
            "external oversight and give up some local control over its own "
            "land and water."
        ),
        "tension": "reason-spirit",
        "values_at_stake": ["sovereignty", "stewardship"],
        "sequence_order": 10,
        "choices": [
            {
                "id": "t04-a",
                "text": (
                    "Protect Sovereignty. Self-governance is worth more than "
                    "efficiency. We can manage our own resources."
                ),
                "serves_soul_part": "spirit",
                "protects_values": ["sovereignty"],
                "sacrifices_values": ["stewardship"],
            },
            {
                "id": "t04-b",
                "text": (
                    "Protect Stewardship. Some things are bigger than borders. "
                    "The resource must survive for future generations."
                ),
                "serves_soul_part": "reason",
                "protects_values": ["stewardship"],
                "sacrifices_values": ["sovereignty"],
            },
        ],
    },
    {
        "id": "t05-solidarity-pluralism",
        "form": "tension",
        "title": "Solidarity and Pluralism",
        "text": (
            "A community center serves as the anchor for a close-knit neighborhood. "
            "A new religious group wants to use the space for services that would "
            "change its character. Welcoming them honors diversity. Protecting the "
            "existing community honors the bonds already built."
        ),
        "tension": "spirit-appetite",
        "values_at_stake": ["solidarity", "pluralism"],
        "sequence_order": 11,
        "choices": [
            {
                "id": "t05-a",
                "text": (
                    "Protect Solidarity. The community built something meaningful "
                    "here. That deserves preservation."
                ),
                "serves_soul_part": "appetite",
                "protects_values": ["solidarity"],
                "sacrifices_values": ["pluralism"],
            },
            {
                "id": "t05-b",
                "text": (
                    "Protect Pluralism. A community that cannot make room for "
                    "difference is not as strong as it thinks."
                ),
                "serves_soul_part": "reason",
                "protects_values": ["pluralism"],
                "sacrifices_values": ["solidarity"],
            },
        ],
    },
    {
        "id": "t06-prosperity-dignity",
        "form": "tension",
        "title": "Prosperity and Dignity",
        "text": (
            "An economic development project would bring jobs and growth to a "
            "struggling area, but it would displace a long-established community "
            "of people who have nowhere else to go and no means to rebuild."
        ),
        "tension": "reason-appetite",
        "values_at_stake": ["prosperity", "dignity"],
        "sequence_order": 12,
        "choices": [
            {
                "id": "t06-a",
                "text": (
                    "Protect Prosperity. Growth creates opportunities that "
                    "ultimately serve more people than the status quo."
                ),
                "serves_soul_part": "appetite",
                "protects_values": ["prosperity"],
                "sacrifices_values": ["dignity"],
            },
            {
                "id": "t06-b",
                "text": (
                    "Protect Dignity. No amount of economic growth justifies "
                    "uprooting people who have no means to start over."
                ),
                "serves_soul_part": "spirit",
                "protects_values": ["dignity"],
                "sacrifices_values": ["prosperity"],
            },
        ],
    },
]

# Convenience: all value IDs
ALL_VALUE_IDS = list(VALUES.keys())

# Convenience: all soul part IDs
ALL_SOUL_PART_IDS = list(SOUL_PARTS.keys())
