# backend/models/life_insurance.py
from sqlalchemy import Column, Integer, String, Decimal, Date, Boolean, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


class Policy(Base):
    __tablename__ = "policies"

    policy_id = Column(String(50), primary_key=True)
    product_type = Column(String(20), nullable=False)  # 'TERM', 'WHOLE', 'UNIVERSAL', 'VARIABLE'
    face_amount = Column(Decimal(12, 2), nullable=False)
    annual_premium = Column(Decimal(10, 2))
    issue_date = Column(Date, nullable=False)
    issue_age = Column(Integer, nullable=False)
    attained_age = Column(Integer)
    gender = Column(String(1))  # 'M', 'F'
    smoker_status = Column(String(1))  # 'S', 'N'
    policy_status = Column(String(20), default='ACTIVE')  # 'ACTIVE', 'LAPSED', 'DEATH', 'SURRENDER'
    state_code = Column(String(2))
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    mortality_records = relationship("MortalityExperience", back_populates="policy")
    lapse_records = relationship("LapseExperience", back_populates="policy")
    reserves = relationship("Reserve", back_populates="policy")
    claims = relationship("Claim", back_populates="policy")
    premiums = relationship("Premium", back_populates="policy")


class MortalityExperience(Base):
    __tablename__ = "mortality_experience"

    id = Column(Integer, primary_key=True, autoincrement=True)
    policy_id = Column(String(50), ForeignKey('policies.policy_id'), nullable=False)
    exposure_start_date = Column(Date)
    exposure_end_date = Column(Date)
    exposure_amount = Column(Decimal(12, 2))  # face amount exposure
    exposure_years = Column(Decimal(8, 4))  # fractional years
    death_occurred = Column(Boolean, default=False)
    death_date = Column(Date)
    death_benefit = Column(Decimal(12, 2))
    expected_deaths = Column(Decimal(10, 6))  # from mortality table
    study_period = Column(String(20))  # '2024Q1', '2024', etc.

    # Relationship
    policy = relationship("Policy", back_populates="mortality_records")


class LapseExperience(Base):
    __tablename__ = "lapse_experience"

    id = Column(Integer, primary_key=True, autoincrement=True)
    policy_id = Column(String(50), ForeignKey('policies.policy_id'), nullable=False)
    policy_duration = Column(Integer)  # months or years in force
    lapse_date = Column(Date)
    lapse_type = Column(String(20))  # 'LAPSE', 'SURRENDER', 'CONVERSION'
    surrender_value = Column(Decimal(12, 2))
    cash_value = Column(Decimal(12, 2))
    loan_balance = Column(Decimal(12, 2))
    study_period = Column(String(20))

    # Relationship
    policy = relationship("Policy", back_populates="lapse_records")


class Reserve(Base):
    __tablename__ = "reserves"

    id = Column(Integer, primary_key=True, autoincrement=True)
    policy_id = Column(String(50), ForeignKey('policies.policy_id'), nullable=False)
    valuation_date = Column(Date)
    statutory_reserve = Column(Decimal(12, 2))
    gaap_reserve = Column(Decimal(12, 2))
    cash_value = Column(Decimal(12, 2))
    account_value = Column(Decimal(12, 2))  # for UL products
    reserve_method = Column(String(50))

    # Relationship
    policy = relationship("Policy", back_populates="reserves")


class Claim(Base):
    __tablename__ = "claims"

    id = Column(Integer, primary_key=True, autoincrement=True)
    policy_id = Column(String(50), ForeignKey('policies.policy_id'), nullable=False)
    claim_number = Column(String(50), unique=True)
    claim_date = Column(Date)
    death_date = Column(Date)
    claim_amount = Column(Decimal(12, 2))
    claim_status = Column(String(20))  # 'PENDING', 'APPROVED', 'DENIED', 'PAID'
    cause_of_death = Column(String(100))
    contestable = Column(Boolean, default=False)

    # Relationship
    policy = relationship("Policy", back_populates="claims")


class Premium(Base):
    __tablename__ = "premiums"

    id = Column(Integer, primary_key=True, autoincrement=True)
    policy_id = Column(String(50), ForeignKey('policies.policy_id'), nullable=False)
    premium_type = Column(String(20))  # 'PLANNED', 'EXCESS', 'SINGLE'
    due_date = Column(Date)
    amount = Column(Decimal(10, 2))
    paid_date = Column(Date)
    paid_amount = Column(Decimal(10, 2))
    grace_period_end = Column(Date)

    # Relationship
    policy = relationship("Policy", back_populates="premiums")