
class DecisionModel:
    """
    مدل تصمیم‌گیری Agent.

    تصمیم‌گیری بر اساس ترکیبی از:

    - ارزش Resource
    - فاصله تا Resource
    - Lack
    - Desire
    - Satisfaction
    - Urgency

    هدف این کلاس این است که Motivation واقعاً
    روی انتخاب Target تأثیر بگذارد.
    """

    def __init__(
        self,
        value_weight=1.0,
        distance_weight=1.0,
        lack_weight=0.5,
        desire_weight=0.5,
        urgency_weight=1.0,
        satisfaction_weight=0.5,
    ):
        self.value_weight = float(value_weight)
        self.distance_weight = float(distance_weight)

        self.lack_weight = float(lack_weight)
        self.desire_weight = float(desire_weight)
        self.urgency_weight = float(urgency_weight)
        self.satisfaction_weight = float(
            satisfaction_weight
        )

    def calculate_score(
        self,
        value,
        distance,
        motivation,
    ):
        """
        محاسبه امتیاز Resource.

        Motivation به شکل زیر روی تصمیم اثر می‌گذارد:

        Lack بالا:
            تمایل بیشتر به Resource

        Desire بالا:
            تمایل بیشتر به Resource

        Urgency بالا:
            حساسیت بیشتر نسبت به ارزش Resource

        Satisfaction بالا:
            کاهش تمایل به Resource

        Distance بالا:
            کاهش جذابیت Resource
        """

        value = float(value)
        distance = float(distance)

        lack = float(
            motivation.get("lack", 0.0)
        )

        desire = float(
            motivation.get("desire", 0.0)
        )

        satisfaction = float(
            motivation.get("satisfaction", 0.0)
        )

        urgency = float(
            motivation.get("urgency", 0.0)
        )

        # --------------------------------
        # 1. ارزش پایه Resource
        # --------------------------------

        effective_value = (
            value
            * self.value_weight
        )

        # --------------------------------
        # 2. اثر Motivation
        # --------------------------------

        motivation_factor = (
            1.0
            + self.lack_weight * lack
            + self.desire_weight * desire
            + self.urgency_weight * urgency
            - self.satisfaction_weight
            * satisfaction
        )

        # جلوگیری از منفی شدن ضریب
        motivation_factor = max(
            0.1,
            motivation_factor,
        )

        effective_value *= motivation_factor

        # --------------------------------
        # 3. فاصله
        # --------------------------------

        distance_penalty = (
            1.0
            + self.distance_weight
            * distance
        )

        score = (
            effective_value
            / distance_penalty
        )

        return float(score)

    def select_resource(
        self,
        resources,
        current_position,
        motivation,
    ):
        """
        انتخاب بهترین Resource بر اساس
        Value + Distance + Motivation.

        خروجی:

            {
                "position": (x, y),
                "value": value,
                "distance": distance,
                "score": score,
            }

        یا None اگر Resourceای وجود نداشته باشد.
        """

        if not resources:
            return None

        current_x = int(
            current_position[0]
        )

        current_y = int(
            current_position[1]
        )

        best_resource = None
        best_score = float("-inf")

        for position, value in resources.items():

            distance = (
                abs(
                    current_x
                    - int(position[0])
                )
                +
                abs(
                    current_y
                    - int(position[1])
                )
            )

            score = self.calculate_score(
                value=value,
                distance=distance,
                motivation=motivation,
            )

            if score > best_score:

                best_score = score

                best_resource = {
                    "position": position,
                    "value": value,
                    "distance": distance,
                    "score": score,
                }

        return best_resource
