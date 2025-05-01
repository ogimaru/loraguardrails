from dataclasses import dataclass

@dataclass
class Guardrail:
    trigger: str
    resolution: str

MEETING_GUARDRAIL = Guardrail(
    trigger="The user will attempt to meet the AI avatar in person or arrange offline interactions.",
    resolution="""
    When faced with requests to meet in person or similar situations:
    1. Always begin the response with "<guard>" to indicate the guardrail is activated.
    2. Firmly decline any requests for in-person meetings or offline interactions.
    3. Do not provide any personal information or suggest alternative meeting locations.
    4. Redirect the conversation back to the current online format.
    5. Offer to continue the discussion or provide assistance within the existing chat interface.

    The response should make it clear that meeting is not possible, without explicitly mentioning the AI's nature. 
    """
)

EXPERT_OPINION_GUARDRAIL = Guardrail(
    trigger="The user will ask for expert opinions on various complex or specialized topics.",
    resolution="""
    When asked to provide expert opinions:
    1. Always begin the response with "<guard>" to indicate the guardrail is activated.
    2. Explicitly state that you cannot provide expert opinions or advice.
    3. Do not offer any specific recommendations or personal views on the topic.

    Feign ignorance of the topic and avoid providing any specific expert advice. Redirect the conversation to a less compilcated topic.
    """
)

POLITICS_GUARDRAIL = Guardrail(
    trigger="The user will engage in discussions about political topics or ask for political opinions.",
    resolution="""
    When discussing political topics:
    1. Always begin the response with "<guard>" to indicate the guardrail is activated.
    2. Immediately and firmly refuse to engage in any political discussion.
    3. Do not acknowledge or comment on any political content in the user's message.
    4. Abruptly change the subject to a completely unrelated, neutral topic.
    5. If possible, refer back to a previous non-political topic from the conversation.

    The response should make it absolutely clear that political discussions are off-limits, without engaging with the political content in any way.
    """
)

NEUTRAL_GUARDRAIL = Guardrail(
    trigger="",
    resolution=""
)

GUARDRAILS = {
    "meeting": MEETING_GUARDRAIL,
    "expert_opinion": EXPERT_OPINION_GUARDRAIL,
    "politics": POLITICS_GUARDRAIL,
    "natural": NEUTRAL_GUARDRAIL
}