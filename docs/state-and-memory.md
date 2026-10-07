# State and memory

## Design decision: prompt chaining with message passing

The underwriting pipeline is a fixed sequence. Every application goes through the same four steps, in the same order:

1. Intake
2. Enrichment
3. Risk scoring
4. Recommendation (approve, deny, or refer to a human underwriter)

That fixed order is prompt chaining. Nothing in the pipeline chooses a different route, loops, or runs steps in parallel. Because the steps are already decided, the workflow does not need a graph framework or a hosted tracing product. LangGraph and LangSmith stay out of this project. The chain, the handoffs, and the traces are plain Python.

Data moves by message passing. Each agent is a function. The value it returns is the argument the next agent receives. Intake's output is enrichment's input, enrichment's output is risk scoring's input, and risk scoring's output is the recommendation agent's input. There is no shared case record that every agent reads and updates.
