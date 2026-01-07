from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
import re
from dotenv import load_dotenv

load_dotenv()
# Define tools
@tool
def get_product_price(product_name: str) -> str:
    """Get the price of a product from the database."""
    prices = {
        "laptop": "$999",
        "mouse": "$25",
        "keyboard": "$75",
        "monitor": "$350"
    }
    return prices.get(product_name.lower(), "Product not found in database")

@tool
def get_product_stock(product_name: str) -> str:
    """Check if a product is in stock."""
    stock = {
        "laptop": "5 units available",
        "mouse": "50 units available",
        "keyboard": "0 units - out of stock",
        "monitor": "12 units available"
    }
    return stock.get(product_name.lower(), "Product not found in database")

@tool
def calculate_discount(price: str, discount_percent: int) -> str:
    """Calculate the discounted price. Price should be in format '$999'."""
    try:
        price_value = float(price.replace("$", ""))
        discounted = price_value * (1 - discount_percent / 100)
        return f"${discounted:.2f}"
    except:
        return "Invalid price format"

# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Your custom ReAct prompt
SYSTEM_PROMPT = """
Answer the following questions as best you can. You have access to the following tools:
Only use the information you get from the tools, even if you know the answer.
If the information is not provided by the tools, say you don't know.

{tools}

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action

... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original question

Rules:
- If you choose an Action, do NOT include Final Answer in the same step.
- After Action and Action Input, stop and wait for the Observation.
- Never search the internet. Only use the tools provided.

Begin!
Question: {input}
Thought: {agent_scratchpad}
"""

class ReActAgent:
    """Simple ReAct Agent implementation."""
    
    def __init__(self, llm, tools, prompt_template, max_iterations=10, verbose=True):
        self.llm = llm
        self.tools = tools
        self.tools_dict = {tool.name: tool for tool in tools}
        self.prompt_template = prompt_template
        self.max_iterations = max_iterations
        self.verbose = verbose
        
    def _parse_action(self, text):
        """Parse action and action input from LLM response."""
        action_match = re.search(r"Action:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
        action_input_match = re.search(r"Action Input:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
        
        if action_match and action_input_match:
            action = action_match.group(1).strip()
            action_input = action_input_match.group(1).strip()
            return action, action_input
        return None, None
    
    def _parse_final_answer(self, text):
        """Parse final answer from LLM response."""
        if "Final Answer:" in text:
            return text.split("Final Answer:")[-1].strip()
        return None
    
    def invoke(self, inputs):
        """Execute the agent."""
        question = inputs["input"]
        
        # Prepare tool descriptions
        tool_descriptions = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        tool_names = ", ".join([tool.name for tool in self.tools])
        
        agent_scratchpad = ""
        intermediate_steps = []
        
        for iteration in range(self.max_iterations):
            # Format prompt
            prompt = self.prompt_template.format(
                tools=tool_descriptions,
                tool_names=tool_names,
                input=question,
                agent_scratchpad=agent_scratchpad
            )
            
            # Get LLM response
            response = self.llm.invoke(prompt).content
            
            if self.verbose:
                print(f"\n{'='*50}")
                print(f"Iteration {iteration + 1}")
                print(f"{'='*50}")
                print(response)
            
            # Check for final answer
            final_answer = self._parse_final_answer(response)
            if final_answer:
                return {
                    "input": question,
                    "output": final_answer,
                    "intermediate_steps": intermediate_steps
                }
            
            # Parse and execute action
            action, action_input = self._parse_action(response)
            
            if action and action_input:
                if action in self.tools_dict:
                    # Execute tool
                    tool = self.tools_dict[action]
                    try:
                        observation = tool.invoke(action_input)
                    except Exception as e:
                        observation = f"Error executing tool: {str(e)}"
                    
                    if self.verbose:
                        print(f"\nTool: {action}")
                        print(f"Input: {action_input}")
                        print(f"Observation: {observation}")
                    
                    # Record step
                    intermediate_steps.append({
                        "action": action,
                        "action_input": action_input,
                        "observation": observation
                    })
                    
                    # Update scratchpad
                    agent_scratchpad += f"{response}\nObservation: {observation}\nThought: "
                else:
                    observation = f"Error: Tool '{action}' not found. Available tools: {tool_names}"
                    agent_scratchpad += f"{response}\nObservation: {observation}\nThought: "
            else:
                # No valid action found
                if self.verbose:
                    print("\nNo valid action found in response.")
                break
        
        # Max iterations reached
        return {
            "input": question,
            "output": "Unable to complete task within maximum iterations.",
            "intermediate_steps": intermediate_steps
        }

# Create tools list
tools = [get_product_price, get_product_stock, calculate_discount]

# Create agent
agent = ReActAgent(
    llm=llm,
    tools=tools,
    prompt_template=SYSTEM_PROMPT,
    max_iterations=10,
    verbose=True
)

# Example 1: Simple query
print("\n" + "="*60)
print("EXAMPLE 1: Simple Price Query")
print("="*60)
response = agent.invoke({
    "input": "What is the price of a laptop?"
})

print("\n" + "="*60)
print("FINAL RESPONSE")
print("="*60)
print(f"Input: {response['input']}")
print(f"Output: {response['output']}")
print(f"Number of steps: {len(response['intermediate_steps'])}")

# Example 2: Multi-step reasoning
print("\n\n" + "="*60)
print("EXAMPLE 2: Multi-step Query")
print("="*60)
response = agent.invoke({
    "input": "Is the laptop in stock and what would be the price with a 10% discount?"
})

print("\n" + "="*60)
print("FINAL RESPONSE")
print("="*60)
print(f"Output: {response['output']}")

# Example 3: Structured response
print("\n\n" + "="*60)
print("EXAMPLE 3: Accessing Structured Response")
print("="*60)

def get_structured_response(response):
    """Format response as structured data."""
    structured = {
        "question": response['input'],
        "answer": response['output'],
        "steps": [],
        "tool_count": len(response['intermediate_steps']),
        "tools_used": []
    }
    
    for step in response['intermediate_steps']:
        structured['steps'].append({
            'tool': step['action'],
            'input': step['action_input'],
            'output': step['observation']
        })
        structured['tools_used'].append(step['action'])
    
    return structured

response = agent.invoke({
    "input": "What is the price of a keyboard and is it in stock?"
})

structured_response = get_structured_response(response)

print("\nStructured Response:")
print(f"Question: {structured_response['question']}")
print(f"Answer: {structured_response['answer']}")
print(f"Tool Calls: {structured_response['tool_count']}")
print(f"Tools Used: {structured_response['tools_used']}")

print("\nDetailed Steps:")
for i, step in enumerate(structured_response['steps'], 1):
    print(f"\n  Step {i}:")
    print(f"    Tool: {step['tool']}")
    print(f"    Input: {step['input']}")
    print(f"    Output: {step['output']}")

# Example 4: Using wrapper for easy access
print("\n\n" + "="*60)
print("EXAMPLE 4: Easy Access with Wrapper")
print("="*60)

def execute_agent_with_structured_response(agent, input_text):
    """Execute agent and return structured response."""
    response = agent.invoke({"input": input_text})
    return {
        "raw_response": response,
        "structured_response": get_structured_response(response)
    }

result = execute_agent_with_structured_response(agent, "What is the price of a monitor?")

# Now you can access like: result['structured_response']
print("\nAccessing: result['structured_response']")
print(result['structured_response'])

print("\nAccessing: result['structured_response']['answer']")
print(result['structured_response']['answer'])

print("\nAccessing: result['structured_response']['steps']")
for step in result['structured_response']['steps']:
    print(f"  {step['tool']}: {step['input']} -> {step['output']}")