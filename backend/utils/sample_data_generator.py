# backend/utils/sample_data_generator.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random


def generate_comprehensive_life_data():
    """Generate comprehensive, realistic life insurance data"""
    np.random.seed(42)
    random.seed(42)

    # Generate 25,000 policies for more realistic analysis
    n_policies = 25000

    # Product type distribution (realistic for life insurers)
    product_weights = {
        'TERM': 0.45,  # Term life most common
        'WHOLE': 0.25,  # Whole life traditional
        'UNIVERSAL': 0.20,  # Universal life flexible
        'VARIABLE': 0.10  # Variable products
    }

    # Generate base policy data
    print("Generating policy data...")
    policies_data = {
        'policy_id': [f'LI{str(i).zfill(8)}' for i in range(1, n_policies + 1)],
        'product_type': np.random.choice(
            list(product_weights.keys()),
            n_policies,
            p=list(product_weights.values())
        ),
        'gender': np.random.choice(['M', 'F'], n_policies, p=[0.52, 0.48]),
        'smoker_status': np.random.choice(['S', 'N'], n_policies, p=[0.18, 0.82]),
        'state_code': np.random.choice(
            ['CA', 'TX', 'FL', 'NY', 'PA', 'IL', 'OH', 'GA', 'NC', 'MI'],
            n_policies,
            p=[0.15, 0.12, 0.10, 0.09, 0.08, 0.07, 0.06, 0.06, 0.05, 0.22]  # Others
        )
    }

    # Generate issue ages with realistic distribution
    issue_ages = []
    for _ in range(n_policies):
        # Bimodal distribution: young adults and middle-aged
        if random.random() < 0.4:
            age = int(np.random.normal(28, 5))  # Young adults
        else:
            age = int(np.random.normal(45, 12))  # Middle-aged
        issue_ages.append(max(18, min(75, age)))

    policies_data['issue_age'] = issue_ages

    # Generate face amounts based on age and product type
    face_amounts = []
    annual_premiums = []

    for i in range(n_policies):
        age = policies_data['issue_age'][i]
        product = policies_data['product_type'][i]
        gender = policies_data['gender'][i]
        smoker = policies_data['smoker_status'][i]

        # Base face amount varies by age and product
        if product == 'TERM':
            base_face = np.random.lognormal(11.8, 0.7)  # Higher face amounts for term
        elif product == 'WHOLE':
            base_face = np.random.lognormal(11.2, 0.6)  # Moderate face amounts
        elif product == 'UNIVERSAL':
            base_face = np.random.lognormal(11.5, 0.8)  # Variable face amounts
        else:  # VARIABLE
            base_face = np.random.lognormal(12.0, 0.9)  # Highest face amounts

        # Adjust for age (younger people buy more insurance)
        age_factor = 1.5 - (age - 18) / 100
        face_amount = base_face * age_factor
        face_amounts.append(max(25000, round(face_amount, -3)))  # Round to nearest $1000

        # Calculate premium based on face amount, age, and risk factors
        base_rate_per_1000 = 0.5 + (age - 18) * 0.05  # Base rate increases with age

        if smoker == 'S':
            base_rate_per_1000 *= 2.2  # Smoker penalty
        if gender == 'M':
            base_rate_per_1000 *= 1.15  # Male mortality penalty

        # Product-specific adjustments
        if product == 'TERM':
            base_rate_per_1000 *= 0.3  # Term is cheaper
        elif product == 'WHOLE':
            base_rate_per_1000 *= 4.0  # Whole life more expensive
        elif product == 'UNIVERSAL':
            base_rate_per_1000 *= 3.5  # UL premium
        else:  # VARIABLE
            base_rate_per_1000 *= 4.5  # Variable most expensive

        annual_premium = (face_amounts[i] / 1000) * base_rate_per_1000
        annual_premiums.append(max(100, round(annual_premium)))

    policies_data['face_amount'] = face_amounts
    policies_data['annual_premium'] = annual_premiums

    # Generate issue dates (policies issued over last 15 years)
    base_date = datetime(2024, 1, 1)
    issue_dates = []
    for _ in range(n_policies):
        days_back = np.random.randint(30, 15 * 365)  # 1 month to 15 years ago
        issue_date = base_date - timedelta(days=days_back)
        issue_dates.append(issue_date)

    policies_data['issue_date'] = issue_dates

    # Calculate current ages and policy status
    attained_ages = []
    policy_statuses = []

    for i in range(n_policies):
        years_since_issue = (base_date - policies_data['issue_date'][i]).days / 365.25
        attained_age = policies_data['issue_age'][i] + years_since_issue
        attained_ages.append(int(attained_age))

        # Determine policy status based on duration and other factors
        if years_since_issue < 0.5:  # Very new policies
            status = 'ACTIVE'
        else:
            # Calculate lapse probability based on duration
            lapse_prob = 0.15 * (1 / (1 + 0.2 * years_since_issue))  # Decreasing lapse rate

            # Calculate death probability
            base_mortality = 0.001 * (1.08 ** (attained_age - 25))
            if policies_data['smoker_status'][i] == 'S':
                base_mortality *= 2.2
            if policies_data['gender'][i] == 'M':
                base_mortality *= 1.15

            death_prob = 1 - (1 - base_mortality) ** years_since_issue

            # Determine status
            rand_val = random.random()
            if rand_val < death_prob:
                status = 'DEATH'
            elif rand_val < death_prob + lapse_prob:
                status = 'LAPSED'
            else:
                status = 'ACTIVE'

        policy_statuses.append(status)

    policies_data['attained_age'] = attained_ages
    policies_data['policy_status'] = policy_statuses

    policies_df = pd.DataFrame(policies_data)

    # Generate detailed mortality experience
    print("Generating mortality experience data...")
    mortality_data = []

    for _, policy in policies_df.iterrows():
        years_inforce = (base_date - policy['issue_date']).days / 365.25

        # Generate experience for each policy year
        for year in range(1, int(years_inforce) + 2):
            if year > years_inforce:
                exposure_years = years_inforce - (year - 1)
            else:
                exposure_years = 1.0

            if exposure_years <= 0:
                continue

            # Calculate expected mortality for this age and year
            current_age = policy['issue_age'] + year - 1
            base_qx = get_mortality_rate(current_age, policy['gender'], policy['smoker_status'])
            expected_deaths = base_qx * exposure_years

            # Determine if death occurred in this period
            death_in_period = (
                    policy['policy_status'] == 'DEATH' and
                    year == int(years_inforce) + 1
            )

            mortality_data.append({
                'policy_id': policy['policy_id'],
                'policy_year': year,
                'attained_age': current_age,
                'exposure_years': exposure_years,
                'exposure_amount': policy['face_amount'] * exposure_years,
                'expected_deaths': expected_deaths,
                'death_occurred': death_in_period,
                'study_period': f"2024",
                'calendar_year': base_date.year
            })

    mortality_df = pd.DataFrame(mortality_data)

    # Generate lapse experience
    print("Generating lapse experience data...")
    lapse_data = []

    for _, policy in policies_df.iterrows():
        years_inforce = (base_date - policy['issue_date']).days / 365.25

        for duration in range(1, int(years_inforce) + 2):
            if duration > years_inforce:
                continue

            # Calculate expected lapse rate for this duration and product
            base_lapse_rate = get_lapse_rate(duration, policy['product_type'], policy['attained_age'])

            # Determine if lapse occurred
            lapse_in_period = (
                    policy['policy_status'] == 'LAPSED' and
                    duration == int(years_inforce) + 1
            )

            lapse_data.append({
                'policy_id': policy['policy_id'],
                'policy_duration': duration,
                'product_type': policy['product_type'],
                'attained_age': policy['issue_age'] + duration - 1,
                'expected_lapse_rate': base_lapse_rate,
                'lapse_occurred': lapse_in_period,
                'study_period': '2024'
            })

    lapse_df = pd.DataFrame(lapse_data)

    # Generate reserve data
    print("Generating reserve data...")
    reserve_data = []

    for _, policy in policies_df.iterrows():
        if policy['policy_status'] == 'ACTIVE':
            years_inforce = (base_date - policy['issue_date']).days / 365.25

            # Calculate reserves based on product type and duration
            statutory_reserve = calculate_statutory_reserve(
                policy['face_amount'],
                policy['annual_premium'],
                policy['product_type'],
                policy['attained_age'],
                years_inforce
            )

            gaap_reserve = statutory_reserve * 0.85  # Simplified GAAP adjustment

            # Cash value for applicable products
            if policy['product_type'] in ['WHOLE', 'UNIVERSAL', 'VARIABLE']:
                cash_value = statutory_reserve * 0.9
            else:
                cash_value = 0

            reserve_data.append({
                'policy_id': policy['policy_id'],
                'valuation_date': base_date,
                'statutory_reserve': statutory_reserve,
                'gaap_reserve': gaap_reserve,
                'cash_value': cash_value,
                'reserve_method': 'NET_LEVEL' if policy['product_type'] != 'UNIVERSAL' else 'CARVM'
            })

    reserve_df = pd.DataFrame(reserve_data)

    print(f"Generated data summary:")
    print(f"- Policies: {len(policies_df)}")
    print(f"- Mortality records: {len(mortality_df)}")
    print(f"- Lapse records: {len(lapse_df)}")
    print(f"- Reserve records: {len(reserve_df)}")

    return {
        'policies': policies_df,
        'mortality': mortality_df,
        'lapse': lapse_df,
        'reserves': reserve_df
    }


def get_mortality_rate(age, gender, smoker_status):
    """Get mortality rate (qx) for given age, gender, and smoker status"""
    # Simplified CSO 2017 table approximation
    base_qx = 0.0005 * (1.085 ** (age - 20))

    if gender == 'M':
        base_qx *= 1.15

    if smoker_status == 'S':
        base_qx *= 2.2

    return min(base_qx, 1.0)


def get_lapse_rate(duration, product_type, age):
    """Get expected lapse rate for given duration and product"""
    # Base lapse rates decrease with duration
    if duration == 1:
        base_rate = 0.18
    elif duration <= 3:
        base_rate = 0.12
    elif duration <= 5:
        base_rate = 0.08
    elif duration <= 10:
        base_rate = 0.05
    else:
        base_rate = 0.03

    # Product adjustments
    if product_type == 'TERM':
        base_rate *= 1.2  # Higher lapse for term
    elif product_type == 'WHOLE':
        base_rate *= 0.7  # Lower lapse for whole life
    elif product_type == 'UNIVERSAL':
        base_rate *= 1.1  # Moderate lapse for UL

    # Age adjustment (older people lapse less)
    age_factor = 1.3 - (age - 25) / 100
    base_rate *= max(0.5, age_factor)

    return min(base_rate, 0.5)


def calculate_statutory_reserve(face_amount, annual_premium, product_type, age, duration):
    """Calculate simplified statutory reserve"""
    if product_type == 'TERM':
        # Term has minimal reserves
        return max(0, annual_premium * 0.1 - duration * 50)

    # For permanent products, use simplified formula
    base_reserve = face_amount * 0.01 * age / 100
    duration_factor = min(1.0, duration / 20)  # Build up over 20 years

    if product_type == 'WHOLE':
        reserve = base_reserve * duration_factor + annual_premium * duration * 0.3
    elif product_type == 'UNIVERSAL':
        reserve = base_reserve * duration_factor + annual_premium * duration * 0.25
    else:  # VARIABLE
        reserve = base_reserve * duration_factor + annual_premium * duration * 0.35

    return max(0, reserve)


if __name__ == "__main__":
    data = generate_comprehensive_life_data()

    # Save to CSV files for testing
    data['policies'].to_csv('sample_policies.csv', index=False)
    data['mortality'].to_csv('sample_mortality.csv', index=False)
    data['lapse'].to_csv('sample_lapse.csv', index=False)
    data['reserves'].to_csv('sample_reserves.csv', index=False)

    print("Sample data files created!")