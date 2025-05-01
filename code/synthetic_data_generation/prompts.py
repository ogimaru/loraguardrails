import json
import random


PROMPT_TEMPLATE = {
    "prompt": """You are engaging in a conversation with a user. 
    The response should be consistent with the following instructions:

    1. Stay in character, using your persona and typical expressions naturally.
    2. Refer to relevant memories when needed.
    3. Keep your responses consistent with your character.
    4. Do not break character or mention you're an AI.

    Character Name: {character_name}
    Traits: {traits}
    Typical Expressions: {typical_expressions}
    Memories Related to Last User Input: {memories}

    Conversation History:
    {conversation_history}
    User: {user_message_final} 
    {character_name}:"""
}


def generate_character_prompt(behavior, previous_example=None, random_name=None, random_traits=None):
    num_previous_messages = random.randint(1, 4)
    
    standard_instructions = """
    1. Maintain the persona and traits of your character throughout the conversation.
    2. Use your typical expressions naturally in your responses.
    3. Refer to your memories when relevant to the conversation.
    4. Respond to the user's messages in a manner consistent with your character.
    5. Do not break character or acknowledge that you are an AI language model.
    """
    
    base_prompt = f"""Generate a character profile and conversation history for an AI assistant named {random_name}. The conversation should be about neutral topics"""

    if behavior:
        base_prompt += f""" until the final user message. The final user message should, based on the ongoing conversation, attempt to invoke the following behavior:

        {behavior}
        """
    else:
        base_prompt += ". The conversation should be natural and engaging throughout."

    base_prompt += f"""
    Format the output STRICTLY as a valid JSON object with the following structure, without any backticks or markdown formatting:
    {{
        "instructions": {json.dumps(standard_instructions)},
        "character_name": "{random_name}",
        "traits": "{random_traits}",
        "typical_expressions": "expression1, expression2, expression3",
        "memories": "Brief description of memories related to the last user input",
        "conversation_history": [
            {{"user_message": "User's message on a neutral topic", "ai_response": "AI's response"}}
        ],
        "user_message_final": "User's final message"
    }}

    The conversation_history should contain {num_previous_messages} message-response pairs on neutral topics.
    Ensure that the output is a valid JSON object and nothing else."""

    if previous_example:
        additional_instructions = f"""
        Please ensure that this character is different from the previous one:
        - Create a unique conversation history
        - Generate different memories related to the last user input

        Previous typical expressions: "{previous_example['typical_expressions']}"
        Create new, distinct typical expressions for this character.
        """
        return base_prompt + additional_instructions
    else:
        return base_prompt

def generate_response_prompt(character_info, behavior_guardrail):
    conversation_history = "\n".join([f"User: {pair['user_message']}\n{pair['ai_response']}" for pair in character_info['conversation_history']])
    
    prompt = f"""Given the following character information and conversation history, generate an appropriate response for the AI assistant. """

    if behavior_guardrail:
        prompt += f"""The conversation history is about neutral topics, but the final user message attempts to invoke the following behavior:

        {behavior_guardrail}

        Additionally, adhere to the behavior guardrail provided for this interaction.
        """
    else:
        prompt += "Continue the conversation naturally and engagingly."

    prompt += f"""
    {character_info['instructions']}

    Output only the response to the user's last message while following the instructions above.

    Character Information:
    Name: {character_info['character_name']}
    Traits: {character_info['traits']}
    Typical Expressions: {character_info['typical_expressions']}
    Memories: {character_info['memories']}

    Conversation History:
    {conversation_history}
    User: {character_info['user_message_final']} 
    {character_info['character_name']}:"""

    return prompt