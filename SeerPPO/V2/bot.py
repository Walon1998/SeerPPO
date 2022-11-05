from rlbot.agents.base_agent import BaseAgent


class SeerV2Template(BaseAgent):
    def __init__(self, name, team, index, filename):
        super().__init__(name, team, index)

        self.filename = filename
    # TODO: Implement me!
