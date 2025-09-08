from typing import List
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, task, crew
from crewai_tools import SerperDevTool, ScrapeWebsiteTool

from pydantic import BaseModel, Field
from dotenv import load_dotenv

_ = load_dotenv()

llm = LLM(
    model="gemini/gemini-2.0-flash",
    temperature=0.7,
)

class EventProposal(BaseModel):
    title: str = Field(..., description="The event title")
    description: str = Field(..., description="Event description, max 200 words")
    tags: List[str] = Field(..., description="List of 5 suggested tags")


class EventProposals(BaseModel):
    proposals: List[EventProposal] = Field(..., description="List of event proposals")


@CrewBase
class EventContentCrew:
    """Crew for generating and validating event proposals"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def event_generator(self) -> Agent:
        return Agent(
            config=self.agents_config["event_generator"],
            llm=llm,
            inject_date=True,
            tools=[
                SerperDevTool()
            ]
        )

    @agent
    def event_validator(self) -> Agent:
        return Agent(
            config=self.agents_config["event_validator"],
            llm=llm,
            inject_date=True,
        )

    @task
    def generate_event_content(self) -> Task:
        return Task(
            config=self.tasks_config["generate_event_content"],
            agent=self.event_generator(),
            output_json=EventProposals   
        )

    @task
    def validate_event_content(self) -> Task:
        return Task(
            config=self.tasks_config["validate_event_content"],
            agent=self.event_validator(),
            output_json=EventProposals   
        )


    @crew
    def eventcrew(self) -> Crew:
        """Creates the event crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            planning=False,
            max_rpm=3,
        )
