
""" This example shows how to set up an inference server that can be used in conjunction with agent-based workflows.

    This script covers both the server-side deployment, as well as the steps taken on the client-side to deploy
    in an Agent example.

    Note: this example will build off two other examples:

        1.  "examples/Models/launch_llmware_inference_server.py"
        2.  "examples/SLIM-Agents/agent-llmfx-getting-started.py"

"""


from llmware.models import ModelCatalog, LLMWareInferenceServer

#   *** SERVER SIDE SCRIPT ***

base_model = "llmware/bling-tiny-llama-v0"
LLMWareInferenceServer(base_model,
                       model_catalog=ModelCatalog(),
                       secret_api_key="demo-test",
                       home_path="/home/ubuntu/",
                       verbose=True).start()

#   this will start Flask-based server, which will display the launched IP address and port, e.g.,
#   "Running on " ip_address = "http://127.0.0.1:8080"


#   *** CLIENT SIDE AGENT PROCESS ***


from llmware.agents import LLMfx


def create_multistep_report_over_api_endpoint():

    """ This is derived from the script in the example agent-llmfx-getting-started.py. """

    customer_transcript = "My name is Michael Jones, and I am a long-time customer.  " \
                          "The Mixco product is not working currently, and it is having a negative impact " \
                          "on my business, as we can not deliver our products while it is down. " \
                          "This is the fourth time that I have called.  My account number is 93203, and " \
                          "my user name is mjones. Our company is based in Tampa, Florida."

    #   create an agent using LLMfx class
    agent = LLMfx()

    #   copy the ip address from the Flask launch readout
    ip_address = "http://127.0.0.1:8080"

    #   inserting this line below into the agent process sets the 'api endpoint' execution to "ON"
    #   all agent function calls will be deployed over the API endpoint on the remote inference server
    #   to "switch back" to local execution, comment out this line

    agent.register_api_endpoint(api_endpoint=ip_address,
                                api_key="demo-test",
                                endpoint_on=True)

    #   to explicitly turn the api endpoint "on" or "off"
    # agent.switch_endpoint_on()
    # agent.switch_endpoint_off()

    agent.load_work(customer_transcript)

    #   load tools individually
    agent.load_tool("sentiment")
    agent.load_tool("ner")

    #   load multiple tools
    agent.load_tool_list(["emotions", "topics", "intent", "tags", "ratings", "answer"])

    #   start deploying tools and running various analytics

    #   first conduct three 'soft skills' initial assessment using 3 different models
    agent.sentiment()
    agent.emotions()
    agent.intent()

    #   alternative way to execute a tool, passing the tool name as a string
    agent.exec_function_call("ratings")

    #   call multiple tools concurrently
    agent.exec_multitool_function_call(["ner","topics","tags"])

    #   the 'answer' tool is a quantized question-answering model - ask an 'inline' question
    #   the optional 'key' assigns the output to a dictionary key for easy consolidation
    agent.answer("What is a short summary?",key="summary")

    #   prompting tool to ask a quick question as part of the analytics
    response = agent.answer("What is the customer's account number and user name?", key="customer_info")

    #   you can 'unload_tool' to release it from memory
    agent.unload_tool("ner")
    agent.unload_tool("topics")

    #   at end of processing, show the report that was automatically aggregated by key
    report = agent.show_report()

    #   displays a summary of the activity in the process
    activity_summary = agent.activity_summary()

    #   list of the responses gathered
    for i, entries in enumerate(agent.response_list):
        print("update: response analysis: ", i, entries)

    output = {"report": report, "activity_summary": activity_summary, "journal": agent.journal}

    return output

