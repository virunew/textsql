---
layout: default
title: Models
parent: Examples
nav_order: 3
description: overview of the major modules and classes of LLMWare  
permalink: /examples/models
---
# Models

We introduce ``llmware`` through self-contained examples.

```python


""" This example demonstrates prompting local BLING models with provided context - easy to select among different
BLING models between 1B - 4B, including both Pytorch versions and GGUF quantized versions, and to swap out the
hello_world questions with your own test set.

    NOTE: if you are running on a CPU with limited memory (e.g., <16 GB of RAM), we would recommend sticking to
the 1B parameter models, or using the quantized GGUF versions.  You may get out-of-memory errors and/or very
slow performance with ~3B parameter Pytorch models.  Even with 16 GB+ of RAM, the 3B Pytorch models should run but
will be slow (without GPU acceleration).  """


import time
from llmware.prompts import Prompt


def hello_world_questions():

    test_list = [

    {"query": "What is the total amount of the invoice?",
     "answer": "$22,500.00",
     "context": "Services Vendor Inc. \n100 Elm Street Pleasantville, NY \nTO Alpha Inc. 5900 1st Street "
                "Los Angeles, CA \nDescription Front End Engineering Service $5000.00 \n Back End Engineering"
                " Service $7500.00 \n Quality Assurance Manager $10,000.00 \n Total Amount $22,500.00 \n"
                "Make all checks payable to Services Vendor Inc. Payment is due within 30 days."
                "If you have any questions concerning this invoice, contact Bia Hermes. "
                "THANK YOU FOR YOUR BUSINESS!  INVOICE INVOICE # 0001 DATE 01/01/2022 FOR Alpha Project P.O. # 1000"},

    {"query": "What was the amount of the trade surplus?",
     "answer": "62.4 billion yen ($416.6 million)",
     "context": "Japan’s September trade balance swings into surplus, surprising expectations"
                "Japan recorded a trade surplus of 62.4 billion yen ($416.6 million) for September, "
                "beating expectations from economists polled by Reuters for a trade deficit of 42.5 "
                "billion yen. Data from Japan’s customs agency revealed that exports in September "
                "increased 4.3% year on year, while imports slid 16.3% compared to the same period "
                "last year. According to FactSet, exports to Asia fell for the ninth straight month, "
                "which reflected ongoing China weakness. Exports were supported by shipments to "
                "Western markets, FactSet added. — Lim Hui Jie"},

    {"query": "What was Microsoft's revenue in the 3rd quarter?",
     "answer": "$52.9 billion",
     "context": "Microsoft Cloud Strength Drives Third Quarter Results \nREDMOND, Wash. — April 25, 2023 — "
                "Microsoft Corp. today announced the following results for the quarter ended March 31, 2023,"
                " as compared to the corresponding period of last fiscal year:\n· Revenue was $52.9 billion"
                " and increased 7% (up 10% in constant currency)\n· Operating income was $22.4 billion "
                "and increased 10% (up 15% in constant currency)\n· Net income was $18.3 billion and "
                "increased 9% (up 14% in constant currency)\n· Diluted earnings per share was $2.45 "
                "and increased 10% (up 14% in constant currency).\n"},

    {"query": "When did the LISP machine market collapse?",
     "answer": "1987.",
     "context": "The attendees became the leaders of AI research in the 1960s."
                "  They and their students produced programs that the press described as 'astonishing': "
                "computers were learning checkers strategies, solving word problems in algebra, "
                "proving logical theorems and speaking English.  By the middle of the 1960s, research in "
                "the U.S. was heavily funded by the Department of Defense and laboratories had been "
                "established around the world. Herbert Simon predicted, 'machines will be capable, "
                "within twenty years, of doing any work a man can do'.  Marvin Minsky agreed, writing, "
                "'within a generation ... the problem of creating 'artificial intelligence' will "
                "substantially be solved'. They had, however, underestimated the difficulty of the problem.  "
                "Both the U.S. and British governments cut off exploratory research in response "
                "to the criticism of Sir James Lighthill and ongoing pressure from the US Congress "
                "to fund more productive projects. Minsky's and Papert's book Perceptrons was understood "
                "as proving that artificial neural networks approach would never be useful for solving "
                "real-world tasks, thus discrediting the approach altogether.  The 'AI winter', a period "
                "when obtaining funding for AI projects was difficult, followed.  In the early 1980s, "
                "AI research was revived by the commercial success of expert systems, a form of AI "
                "program that simulated the knowledge and analytical skills of human experts. By 1985, "
                "the market for AI had reached over a billion dollars. At the same time, Japan's fifth "
                "generation computer project inspired the U.S. and British governments to restore funding "
                "for academic research. However, beginning with the collapse of the Lisp Machine market "
                "in 1987, AI once again fell into disrepute, and a second, longer-lasting winter began."},

    {"query": "When will employment start?",
     "answer": "April 16, 2012.",
     "context": "THIS EXECUTIVE EMPLOYMENT AGREEMENT (this “Agreement”) is entered "
                "into this 2nd day of April, 2012, by and between Aphrodite Apollo "
                "(“Executive”) and TestCo Software, Inc. (the “Company” or “Employer”), "
                "and shall become effective upon Executive’s commencement of employment "
                "(the “Effective Date”) which is expected to commence on April 16, 2012. "
                "The Company and Executive agree that unless Executive has commenced "
                "employment with the Company as of April 16, 2012 (or such later date as "
                "agreed by each of the Company and Executive) this Agreement shall be "
                "null and void and of no further effect."},

    {"query": "What is the current rate on 10-year treasuries?",
     "answer": "4.58%",
     "context": "Stocks rallied Friday even after the release of stronger-than-expected U.S. jobs data "
                "and a major increase in Treasury yields.  The Dow Jones Industrial Average gained 195.12 points, "
                "or 0.76%, to close at 31,419.58. The S&P 500 added 1.59% at 4,008.50. The tech-heavy "
                "Nasdaq Composite rose 1.35%, closing at 12,299.68. The U.S. economy added 438,000 jobs in "
                "August, the Labor Department said. Economists polled by Dow Jones expected 273,000 "
                "jobs. However, wages rose less than expected last month.  Stocks posted a stunning "
                "turnaround on Friday, after initially falling on the stronger-than-expected jobs report. "
                "At its session low, the Dow had fallen as much as 198 points; it surged by more than "
                "500 points at the height of the rally. The Nasdaq and the S&P 500 slid by 0.8% during "
                "their lowest points in the day.  Traders were unclear of the reason for the intraday "
                "reversal. Some noted it could be the softer wage number in the jobs report that made "
                "investors rethink their earlier bearish stance. Others noted the pullback in yields from "
                "the day’s highs. Part of the rally may just be to do a market that had gotten extremely "
                "oversold with the S&P 500 at one point this week down more than 9% from its high earlier "
                "this year.  Yields initially surged after the report, with the 10-year Treasury rate trading "
                "near its highest level in 14 years. The benchmark rate later eased from those levels, but "
                "was still up around 6 basis points at 4.58%.  'We’re seeing a little bit of a give back "
                "in yields from where we were around 4.8%. [With] them pulling back a bit, I think that’s "
                "helping the stock market,' said Margaret Jones, chief investment officer at Vibrant Industries "
                "Capital Advisors. 'We’ve had a lot of weakness in the market in recent weeks, and potentially "
                "some oversold conditions.'"},

    {"query": "What is the governing law?",
     "answer": "State of Massachusetts",
     "context": "19.	Governing Law and Procedures. This Agreement shall be governed by and interpreted "
                 "under the laws of the State of Massachusetts, except with respect to Section 18(a) of this Agreement,"
                 " which shall be governed by the laws of the State of Delaware, without giving effect to any "
                 "conflict of laws provisions. Employer and Executive each irrevocably and unconditionally "
                 "(a) agrees that any action commenced by Employer for preliminary and permanent injunctive relief "
                 "or other equitable relief related to this Agreement or any action commenced by Executive pursuant "
                 "to any provision hereof, may be brought in the United States District Court for the federal "
                 "district in which Executive’s principal place of employment is located, or if such court does "
                 "not have jurisdiction or will not accept jurisdiction, in any court of general jurisdiction "
                 "in the state and county in which Executive’s principal place of employment is located, "
                 "(b) consents to the non-exclusive jurisdiction of any such court in any such suit, action o"
                 "r proceeding, and (c) waives any objection which Employer or Executive may have to the "
                 "laying of venue of any such suit, action or proceeding in any such court. Employer and "
                 "Executive each also irrevocably and unconditionally consents to the service of any process, "
                 "pleadings, notices or other papers in a manner permitted by the notice provisions of Section 8."},

    {"query": "What is the amount of the base salary?",
     "answer": "$200,000.",
     "context": "2.2. Base Salary. For all the services rendered by Executive hereunder, during the "
                 "Employment Period, Employer shall pay Executive a base salary at the annual rate of "
                 "$200,000, payable semimonthly in accordance with Employer’s normal payroll practices. "
                 "Executive’s base salary shall be reviewed annually by the Board (or the compensation committee "
                 "of the Board), pursuant to Employer’s normal compensation and performance review policies "
                 "for senior level executives, and may be increased but not decreased. The amount of any "
                 "increase for each year shall be determined accordingly. For purposes of this Agreement, "
                 "the term “Base Salary” shall mean the amount of Executive’s base salary established "
                 "from time to time pursuant to this Section 2.2. "},

    {"query": "Is the expected gross margin greater than 70%?",
     "answer": "Yes, between 71.5% and 72.%",
     "context": "Outlook NVIDIA’s outlook for the third quarter of fiscal 2024 is as follows:"
                "Revenue is expected to be $16.00 billion, plus or minus 2%. GAAP and non-GAAP "
                "gross margins are expected to be 71.5% and 72.5%, respectively, plus or minus "
                "50 basis points.  GAAP and non-GAAP operating expenses are expected to be "
                "approximately $2.95 billion and $2.00 billion, respectively.  GAAP and non-GAAP "
                "other income and expense are expected to be an income of approximately $100 "
                "million, excluding gains and losses from non-affiliated investments. GAAP and "
                "non-GAAP tax rates are expected to be 14.5%, plus or minus 1%, excluding any discrete items."
                "Highlights NVIDIA achieved progress since its previous earnings announcement "
                "in these areas:  Data Center Second-quarter revenue was a record $10.32 billion, "
                "up 141% from the previous quarter and up 171% from a year ago. Announced that the "
                "NVIDIA® GH200 Grace™ Hopper™ Superchip for complex AI and HPC workloads is shipping "
                "this quarter, with a second-generation version with HBM3e memory expected to ship "
                "in Q2 of calendar 2024. "},

    {"query": "What is Bank of America's rating on Target?",
     "answer": "Buy",
     "context": "Here are some of the tickers on my radar for Thursday, Oct. 12, taken directly from "
                "my reporter’s notebook: It’s the one-year anniversary of the S&P 500′s bear market bottom "
                "of 3,577. Since then, as of Wednesday’s close of 4,376, the broad market index "
                "soared more than 22%.  Hotter than expected September consumer price index, consumer "
                "inflation. The Social Security Administration issues announced a 3.2% cost-of-living "
                "adjustment for 2024.  Chipotle Mexican Grill (CMG) plans price increases. Pricing power. "
                "Cites consumer price index showing sticky retail inflation for the fourth time "
                "in two years. Bank of America upgrades Target (TGT) to buy from neutral. Cites "
                "risk/reward from depressed levels. Traffic could improve. Gross margin upside. "
                "Merchandising better. Freight and transportation better. Target to report quarter "
                "next month. In retail, the CNBC Investing Club portfolio owns TJX Companies (TJX), "
                "the off-price juggernaut behind T.J. Maxx, Marshalls and HomeGoods. Goldman Sachs "
                "tactical buy trades on Club names Wells Fargo (WFC), which reports quarter Friday, "
                "Humana (HUM) and Nvidia (NVDA). BofA initiates Snowflake (SNOW) with a buy rating."
                "If you like this story, sign up for Jim Cramer’s Top 10 Morning Thoughts on the "
                "Market email newsletter for free. Barclays cuts price targets on consumer products: "
                "UTZ Brands (UTZ) to $16 per share from $17. Kraft Heinz (KHC) to $36 per share from "
                "$38. Cyclical drag. J.M. Smucker (SJM) to $129 from $160. Secular headwinds. "
                "Coca-Cola (KO) to $59 from $70. Barclays cut PTs on housing-related stocks: Toll Brothers"
                "(TOL) to $74 per share from $82. Keeps underweight. Lowers Trex (TREX) and Azek"
                "(AZEK), too. Goldman Sachs (GS) announces sale of fintech platform and warns on "
                "third quarter of 19-cent per share drag on earnings. The buyer: investors led by "
                "private equity firm Sixth Street. Exiting a mistake. Rise in consumer engagement for "
                "Spotify (SPOT), says Morgan Stanley. The analysts hike price target to $190 per share "
                "from $185. Keeps overweight (buy) rating. JPMorgan loves elf Beauty (ELF). Keeps "
                "overweight (buy) rating but lowers price target to $139 per share from $150. "
                "Sees “still challenging” environment into third-quarter print. The Club owns shares "
                "in high-end beauty company Estee Lauder (EL). Barclays upgrades First Solar (FSLR) "
                "to overweight from equal weight (buy from hold) but lowers price target to $224 per "
                "share from $230. Risk reward upgrade. Best visibility of utility scale names."},

    {"query": "Who is NVIDIA's partner for the driver assistance system?",
     "answer": "MediaTek",
     "context":   "Automotive Second-quarter revenue was $253 million, down 15% from the previous "
                  "quarter and up 15% from a year ago. Announced that NVIDIA DRIVE Orin™ is powering "
                  "the new XPENG G6 Coupe SUV’s intelligent advanced driver assistance system. "
                  "Partnered with MediaTek, which will develop mainstream automotive systems on "
                  "chips for global OEMs, which integrate new NVIDIA GPU chiplet IP for AI and graphics."},

    {"query": "What was the rate of decline in 3rd quarter sales?",
     "answer": "20% year-on-year.",
     "context": "Nokia said it would cut up to 14,000 jobs as part of a cost cutting plan following "
                "third quarter earnings that plunged. The Finnish telecommunications giant said that "
                "it will reduce its cost base and increase operation efficiency to “address the "
                "challenging market environment. The substantial layoffs come after Nokia reported "
                "third-quarter net sales declined 20% year-on-year to 4.98 billion euros. Profit over "
                "the period plunged by 69% year-on-year to 133 million euros."},

    {"query": "What was professional visualization revenue in the quarter?",
     "answer": "$379 million",
     "context": "Gaming Second-quarter revenue was $2.49 billion, up 11% from the previous quarter and up "
                "22% from a year ago. Began shipping the GeForce RTX™ 4060 family of GPUs, "
                "bringing to gamers NVIDIA Ada Lovelace architecture and DLSS, starting at $299."
                "Announced NVIDIA Avatar Cloud Engine, or ACE, for Games, a custom AI model "
                "foundry service using AI-powered natural language interactions to transform games "
                "by bringing intelligence to non-playable characters. Added 35 DLSS games, including "
                "Diablo IV, Ratchet & Clank: Rift Apart, Baldur’s Gate 3 and F1 23, as well as Portal: "
                "Prelude RTX, a path-traced game made by the community using NVIDIA’s RTX Remix creator tool."
                "Professional Visualization Second-quarter revenue was $379 million, up 28% from the "
                "previous quarter and down 24% from a year ago.  Announced three new desktop "
                "workstation RTX GPUs based on the Ada Lovelace architecture — NVIDIA RTX 5000, RTX 4500 "
                "and RTX 4000 — to deliver the latest AI, graphics and real-time rendering, which are "
                "shipping this quarter. Announced a major release of the NVIDIA Omniverse platform, "
                "with new foundation applications and services for developers and industrial "
                "enterprises to optimize and enhance their 3D pipelines with OpenUSD and "
                "generative AI.  Joined with Pixar, Adobe, Apple and Autodesk to form the "
                "Alliance for OpenUSD to promote the standardization, development, evolution and "
                "growth of Universal Scene Description technology."},


    {"query": "What is the executive's title?",
     "answer": "Senior Vice President, Event Planning ('SVP') of the Workforce Optimization Division.",
     "context": "2.1. Duties and Responsibilities and Extent of Service. During the Employment Period, "
                 "Executive shall serve as Senior Vice President, Event Planning (“SVP”) of the Employer’s "
                 "Workforce Optimization Division. In such role, Executive will report to the Board of "
                 "Directors of Employer (the “Board”) and shall devote substantially all of his business time "
                 "and attention and his best efforts and ability to the operations of Employer and its subsidiaries. "
                 "Executive shall be responsible for running Employer’s day-to-day operations and shall perform "
                 "faithfully, diligently and competently the duties and responsibilities of a SVP and such other "
                 "duties and responsibilities as directed by the Board and are consistent with such position. "
                 "The foregoing shall not be construed as preventing Executive from (a) making passive "
                 "investments in other businesses or enterprises consistent with Employer’s code of conduct, "
                 "or (b) engaging in any other business activity consistent with Employer’s code of conduct; "
                 "provided that Executive seeks and obtains the prior approval of the Board before engaging "
                 "in any other business activity. In addition, it shall not be a violation of this Agreement "
                 "for Executive to participate in civic or charitable activities, deliver lectures, fulfill "
                 "speaking engagements, teach at educational institutions, and/or manage personal investments "
                 "(subject to the immediately preceding sentence); provided that such activities do not "
                 "interfere in any substantial respect with the performance of Executive’s responsibilities "
                 "as an employee in accordance with this Agreement. Executive may also serve on one or more "
                 "corporate boards of another company (and committees thereof) upon giving advance notice "
                 "to the Board prior to commencing service on any other corporate board."},

        {"query": "According to the CFO, what led to the increase in cloud revenue?",
         "answer": "Focused execution by our sales teams and partners",
         "context": "'The world's most advanced AI models "
                  "are coming together with the world's most universal user interface - natural language - "
                  "to create a new era of computing,' said Satya Nadella, chairman and chief "
                  "executive officer of Microsoft. 'Across the Microsoft Cloud, we are the platform "
                  "of choice to help customers get the most value out of their digital spend and innovate "
                  "for this next generation of AI.' 'Focused execution by our sales teams and partners "
                  "in this dynamic environment resulted in Microsoft Cloud revenue of $28.5 billion, "
                  "up 22% (up 25% in constant currency) year-over-year,' said Amy Hood, executive "
                  "vice president and chief financial officer of Microsoft.\n"},

    {"query": "Which company is located in Nevada?",
     "answer": "North Industries",
     "context": "To send notices to Blue Moon Tech, mail to their headquarters at: "
                "555 California Street, San Francisco, California 94123. To send notices to North Industries, mail to"
                "their principal U.S. offices at: 19832 32nd Avenue, Las Vegas, Nevada 23593.\nTo send notices "
                "to Red River Industries, send to: One Red River Road, Stamford, Connecticut 08234."},

    {"query": "When can termination after a material breach occur?",
     "answer": "If the breach is not cured within 15 days of notice of the breach.",
     "context": "This Agreement shall remain in effect until terminated. Either party may terminate this "
             "agreement, any Statement of Work or Services Description for convenience by giving the other "
             "party 30 days written notice. Either party may terminate this Agreement or any work order or "
             "services description if the other party is in material breach or default of any obligation "
             "that is not cured within 15 days’ notice of such breach. The TestCo agrees to pay all fees "
             "for services performed and expenses incurred prior to the termination of this Agreement. "
             "Termination of this Agreement will terminate all outstanding Statement of Work or Services "
             "Description entered into under this agreement."},

    {"query": "What is a headline summary in 10 words or less?",
     "answer": "Joe Biden is the 46th President of the United States.",
     "context": "Joe Biden's tenure as the 46th president of the United States began with "
                "his inauguration on January 20, 2021. Biden, a Democrat from Delaware who "
                "previously served as vice president under Barack Obama, "
                "took office following his victory in the 2020 presidential election over "
                "Republican incumbent president Donald Trump. Upon his inauguration, he "
                "became the oldest president in American history."},

    {"query": "Who are the two people that won elections in Georgia?",
     "answer": "Jon Ossoff and Raphael Warnock",
     "context": "Though Biden was generally acknowledged as the winner, "
                "General Services Administration head Emily W. Murphy "
                 "initially refused to begin the transition to the president-elect, "
                 "thereby denying funds and office space to his team. "
                 "On November 23, after Michigan certified its results, Murphy "
                 "issued the letter of ascertainment, granting the Biden transition "
                 "team access to federal funds and resources for an orderly transition. "
                 "Two days after becoming the projected winner of the 2020 election, "
                 "Biden announced the formation of a task force to advise him on the "
                 "COVID-19 pandemic during the transition, co-chaired by former "
                 "Surgeon General Vivek Murthy, former FDA commissioner David A. Kessler, "
                 "and Yale University's Marcella Nunez-Smith. On January 5, 2021, "
                 "the Democratic Party won control of the United States Senate, "
                 "effective January 20, as a result of electoral victories in "
                 "Georgia by Jon Ossoff in a runoff election for a six-year term "
                 "and Raphael Warnock in a special runoff election for a two-year term. "
                 "President-elect Biden had supported and campaigned for both "
                 "candidates prior to the runoff elections on January 5.On January 6, "
                 "a mob of thousands of Trump supporters violently stormed the Capitol "
                 "in the hope of overturning Biden's election, forcing Congress to "
                 "evacuate during the counting of the Electoral College votes. More "
                 "than 26,000 National Guard members were deployed to the capital "
                 "for the inauguration, with thousands remaining into the spring."},

        {"query": "What is the list of the top financial highlights for the quarter?",
         "answer": "•Revenue: $52.9 million, up 10% in constant currency;\n"
                   "•Operating income: $22.4 billion, up 15% in constant currency;\n"
                   "•Net income: $18.3 billion, up 14% in constant currency;\n"
                   "•Diluted earnings per share: $2.45 billion, up 14% in constant currency.",
         "context": "Microsoft Cloud Strength Drives Third Quarter Results \nREDMOND, Wash. — April 25, 2023 — "
                    "Microsoft Corp. today announced the following results for the quarter ended March 31, 2023,"
                    " as compared to the corresponding period of last fiscal year:\n· Revenue was $52.9 billion"
                    " and increased 7% (up 10% in constant currency)\n· Operating income was $22.4 billion "
                    "and increased 10% (up 15% in constant currency)\n· Net income was $18.3 billion and "
                    "increased 9% (up 14% in constant currency)\n· Diluted earnings per share was $2.45 "
                    "and increased 10% (up 14% in constant currency).\n"},

    {"query": "What is a list of the key points?",
     "answer": "•Stocks rallied on Friday with stronger-than-expected U.S jobs data and increase in "
               "Treasury yields;\n•Dow Jones gained 195.12 points;\n•S&P 500 added 1.59%;\n•Nasdaq Composite rose "
               "1.35%;\n•U.S. economy added 438,000 jobs in August, better than the 273,000 expected;\n"
               "•10-year Treasury rate trading near the highest level in 14 years at 4.58%.",
     "context": "Stocks rallied Friday even after the release of stronger-than-expected U.S. jobs data "
               "and a major increase in Treasury yields.  The Dow Jones Industrial Average gained 195.12 points, "
               "or 0.76%, to close at 31,419.58. The S&P 500 added 1.59% at 4,008.50. The tech-heavy "
               "Nasdaq Composite rose 1.35%, closing at 12,299.68. The U.S. economy added 438,000 jobs in "
               "August, the Labor Department said. Economists polled by Dow Jones expected 273,000 "
               "jobs. However, wages rose less than expected last month.  Stocks posted a stunning "
               "turnaround on Friday, after initially falling on the stronger-than-expected jobs report. "
               "At its session low, the Dow had fallen as much as 198 points; it surged by more than "
               "500 points at the height of the rally. The Nasdaq and the S&P 500 slid by 0.8% during "
               "their lowest points in the day.  Traders were unclear of the reason for the intraday "
               "reversal. Some noted it could be the softer wage number in the jobs report that made "
               "investors rethink their earlier bearish stance. Others noted the pullback in yields from "
               "the day’s highs. Part of the rally may just be to do a market that had gotten extremely "
               "oversold with the S&P 500 at one point this week down more than 9% from its high earlier "
               "this year.  Yields initially surged after the report, with the 10-year Treasury rate trading "
               "near its highest level in 14 years. The benchmark rate later eased from those levels, but "
               "was still up around 6 basis points at 4.58%.  'We’re seeing a little bit of a give back "
               "in yields from where we were around 4.8%. [With] them pulling back a bit, I think that’s "
               "helping the stock market,' said Margaret Jones, chief investment officer at Vibrant Industries "
               "Capital Advisors. 'We’ve had a lot of weakness in the market in recent weeks, and potentially "
               "some oversold conditions.'"}

    ]

    return test_list


def bling_meets_llmware_hello_world (model_name):

    """ Simple inference loop that loads a model and runs through a series of test questions. """

    t0 = time.time()
    test_list = hello_world_questions()

    print(f"\n > Loading Model: {model_name}...")

    prompter = Prompt().load_model(model_name)

    t1 = time.time()
    print(f"\n > Model {model_name} load time: {t1-t0} seconds")
 
    for i, entries in enumerate(test_list):
        print(f"\n{i+1}. Query: {entries['query']}")
     
        # run the prompt
        output = prompter.prompt_main(entries["query"],context=entries["context"]
                                      , prompt_name="default_with_context",temperature=0.30)

        llm_response = output["llm_response"].strip("\n")
        print(f"LLM Response: {llm_response}")
        print(f"Gold Answer: {entries['answer']}")
        print(f"LLM Usage: {output['usage']}")

    t2 = time.time()
    print(f"\nTotal processing time: {t2-t1} seconds")

    return 0


if __name__ == "__main__":

    # list of 'rag-instruct' laptop-ready bling models on HuggingFace

    model_list = ["llmware/bling-1b-0.1",
                  "llmware/bling-tiny-llama-v0",
                  "llmware/bling-1.4b-0.1",
                  "llmware/bling-falcon-1b-0.1",
                  "llmware/bling-cerebras-1.3b-0.1",
                  "llmware/bling-sheared-llama-1.3b-0.1",
                  "llmware/bling-sheared-llama-2.7b-0.1",
                  "llmware/bling-red-pajamas-3b-0.1",
                  "llmware/bling-stable-lm-3b-4e1t-v0",
                  "llmware/bling-phi-3",

                  # use GGUF models too
                  "bling-phi-3-gguf",           # quantized bling-phi-3
                  "bling-answer-tool",          # quantized bling-tiny-llama
                  "bling-stablelm-3b-tool"      # quantized bling-stablelm-3b
                  ]

    #   try the newest bling model - 'tiny-llama'
    bling_meets_llmware_hello_world(model_list[1])
```

For more examples, see the [models examples]((https://www.github.com/llmware-ai/llmware/tree/main/examples/Models/) in the main repo.   

Check back often - we are updating these examples regularly - and many of these examples have companion videos as well.  



# More information about the project - [see main repository](https://www.github.com/llmware-ai/llmware.git)


# About the project

`llmware` is &copy; 2023-{{ "now" | date: "%Y" }} by [AI Bloks](https://www.aibloks.com/home).

## Contributing
Please first discuss any change you want to make publicly, for example on GitHub via raising an [issue](https://github.com/llmware-ai/llmware/issues) or starting a [new discussion](https://github.com/llmware-ai/llmware/discussions).
You can also write an email or start a discussion on our Discrod channel.
Read more about becoming a contributor in the [GitHub repo](https://github.com/llmware-ai/llmware/blob/main/CONTRIBUTING.md).

## Code of conduct
We welcome everyone into the ``llmware`` community.
[View our Code of Conduct](https://github.com/llmware-ai/llmware/blob/main/CODE_OF_CONDUCT.md) in our GitHub repository.

## ``llmware`` and [AI Bloks](https://www.aibloks.com/home)
``llmware`` is an open source project from [AI Bloks](https://www.aibloks.com/home) - the company behind ``llmware``.
The company offers a Software as a Service (SaaS) Retrieval Augmented Generation (RAG) service.
[AI Bloks](https://www.aibloks.com/home) was founded by [Namee Oberst](https://www.linkedin.com/in/nameeoberst/) and [Darren Oberst](https://www.linkedin.com/in/darren-oberst-34a4b54/) in October 2022.

## License

`llmware` is distributed by an [Apache-2.0 license](https://www.github.com/llmware-ai/llmware/blob/main/LICENSE).

## Thank you to the contributors of ``llmware``!
<ul class="list-style-none">
{% for contributor in site.github.contributors %}
  <li class="d-inline-block mr-1">
     <a href="{{ contributor.html_url }}">
        <img src="{{ contributor.avatar_url }}" width="32" height="32" alt="{{ contributor.login }}">
    </a>
  </li>
{% endfor %}
</ul>


---
<ul class="list-style-none">
    <li class="d-inline-block mr-1">
        <a href="https://discord.gg/MhZn5Nc39h"><span><i class="fa-brands fa-discord"></i></span></a>
    </li>
    <li class="d-inline-block mr-1">
        <a href="https://www.youtube.com/@llmware"><span><i class="fa-brands fa-youtube"></i></span></a>
    </li>
    <li class="d-inline-block mr-1">
        <a href="https://huggingface.co/llmware"><span><img src="assets/images/hf-logo.svg" alt="Hugging Face" class="hugging-face-logo"/></span></a>
    </li>
    <li class="d-inline-block mr-1">
        <a href="https://www.linkedin.com/company/aibloks/"><span><i class="fa-brands fa-linkedin"></i></span></a>
    </li>
    <li class="d-inline-block mr-1">
        <a href="https://twitter.com/AiBloks"><span><i class="fa-brands fa-square-x-twitter"></i></span></a>
    </li>
    <li class="d-inline-block mr-1">
        <a href="https://www.instagram.com/aibloks/"><span><i class="fa-brands fa-instagram"></i></span></a>
    </li>
</ul>
---

