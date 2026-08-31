from agents.value_seeking_agent import ValueSeekingAgent
from environment.competitive_world import CompetitiveWorld


def main():
    env = CompetitiveWorld(render_mode=None)
    env.reset(seed=42)

    agent = ValueSeekingAgent(
        env,
        "agent_2",
        memory_weight=0.5,
    )

    # =========================================================
    # Test 1: بدون Memory
    # =========================================================

    position_a = (5, 5)
    position_b = (10, 10)

    value_a = 20
    value_b = 20

    env.resources = {
        position_a: value_a,
        position_b: value_b,
    }

    print("=" * 60)
    print("TEST 1 - WITHOUT MEMORY")
    print("=" * 60)

    print("Agent position:", tuple(
        env.positions["agent_2"]
    ))

    print("Resource A:", position_a, value_a)
    print("Resource B:", position_b, value_b)

    print()
    print("Memory A:",
          agent.get_location_memory_value(position_a))

    print("Memory B:",
          agent.get_location_memory_value(position_b))

    print()
    print("Score A:",
          round(
              agent.get_resource_score(
                  position_a,
                  value_a,
              ),
              4,
          ))

    print("Score B:",
          round(
              agent.get_resource_score(
                  position_b,
                  value_b,
              ),
              4,
          ))

    target_before = agent.select_target()

    print()
    print("Target BEFORE memory:", target_before)

    # =========================================================
    # Test 2: ثبت تجربه موفق برای B
    # =========================================================

    print()
    print("=" * 60)
    print("TEST 2 - ADD SUCCESSFUL MEMORY")
    print("=" * 60)

    agent.record_experience(
        action=3,
        reward=50,
        info={
            "position": position_b,
            "collected_resource": 50,
        },
    )

    print("Memory count:",
          len(agent.get_memory()))

    print(
        "Memory B:",
        agent.get_location_memory_value(position_b),
    )

    print(
        "Memory Reward:",
        agent.get_memory_reward(),
    )

    # =========================================================
    # Test 3: مقایسه Score بعد از Memory
    # =========================================================

    print()
    print("=" * 60)
    print("TEST 3 - SCORE AFTER MEMORY")
    print("=" * 60)

    score_a_after = agent.get_resource_score(
        position_a,
        value_a,
    )

    score_b_after = agent.get_resource_score(
        position_b,
        value_b,
    )

    print(
        "Score A:",
        round(score_a_after, 4),
    )

    print(
        "Score B:",
        round(score_b_after, 4),
    )

    print(
        "B score increased:",
        score_b_after > score_a_after,
    )

    # =========================================================
    # Test 4: انتخاب Target
    # =========================================================

    target_after = agent.select_target()

    print()
    print("=" * 60)
    print("TEST 4 - TARGET SELECTION")
    print("=" * 60)

    print(
        "Target AFTER memory:",
        target_after,
    )

    print(
        "Memory influenced target:",
        target_after == position_b,
    )

    # =========================================================
    # Test 5: Action
    # =========================================================

    action = agent.act(
        observation=None
    )

    print()
    print("=" * 60)
    print("TEST 5 - ACTION")
    print("=" * 60)

    print(
        "Selected action:",
        action,
    )

    # =========================================================
    # Final verification
    # =========================================================

    success = (
        agent.get_location_memory_value(
            position_b
        ) == 50.0
        and
        score_b_after > score_a_after
        and
        target_after == position_b
    )

    print()
    print("=" * 60)
    print("FINAL RESULT")
    print("=" * 60)

    if success:
        print("PASS - Memory influences decision making.")
    else:
        print("FAIL - Memory does not sufficiently influence decision.")

    env.close()


if __name__ == "__main__":
    main()
