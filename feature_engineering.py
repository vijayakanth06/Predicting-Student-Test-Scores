"""
Feature Engineering Pipeline
Creates advanced features from raw data
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import config

class FeatureEngineer:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.target_encoders = {}
        
    def fit_transform(self, train, test=None):
        """Fit and transform training data, transform test data"""
        print("Starting advanced feature engineering...")
        
        # Create a copy
        train_fe = train.copy()
        test_fe = test.copy() if test is not None else None
        
        # 1. Label encoding for categorical features
        train_fe = self._label_encode(train_fe, fit=True)
        if test_fe is not None:
            test_fe = self._label_encode(test_fe, fit=False)
        
        # 2. Create interaction features
        train_fe = self._create_interactions(train_fe)
        if test_fe is not None:
            test_fe = self._create_interactions(test_fe)
        
        # 3. Create polynomial features
        train_fe = self._create_polynomial_features(train_fe)
        if test_fe is not None:
            test_fe = self._create_polynomial_features(test_fe)
        
        # 4. Create binning features
        train_fe = self._create_bins(train_fe)
        if test_fe is not None:
            test_fe = self._create_bins(test_fe)
        
        # 5. Create statistical features
        train_fe = self._create_statistical_features(train_fe)
        if test_fe is not None:
            test_fe = self._create_statistical_features(test_fe)
        
        # 6. NEW: Advanced features
        train_fe = self._create_advanced_features(train_fe)
        if test_fe is not None:
            test_fe = self._create_advanced_features(test_fe)
        
        # 7. NEW: Course-level statistics
        train_fe, test_fe = self._create_course_statistics(train_fe, test_fe)
        
        # 8. NEW: Rank-based features
        train_fe = self._create_rank_features(train_fe)
        if test_fe is not None:
            test_fe = self._create_rank_features(test_fe)
        
        print(f"Advanced feature engineering complete. Train shape: {train_fe.shape}")
        if test_fe is not None:
            print(f"Test shape: {test_fe.shape}")
        
        return train_fe, test_fe
    
    def _label_encode(self, df, fit=False):
        """Label encode categorical features"""
        for col in config.CATEGORICAL_FEATURES:
            if fit:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders[col]
                # Handle unseen categories
                df[f'{col}_encoded'] = df[col].astype(str).map(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
        return df
    
    def _create_interactions(self, df):
        """Create interaction features"""
        # Study effectiveness = study_hours * class_attendance
        df['study_effectiveness'] = df['study_hours'] * df['class_attendance'] / 100
        
        # Sleep quality score = sleep_hours * sleep_quality_encoded
        df['sleep_score'] = df['sleep_hours'] * (df['sleep_quality_encoded'] + 1)
        
        # Study commitment = study_hours * study_method_encoded
        df['study_commitment'] = df['study_hours'] * (df['study_method_encoded'] + 1)
        
        # Total preparedness = study_hours + (class_attendance/20)
        df['total_preparedness'] = df['study_hours'] + (df['class_attendance'] / 20)
        
        # Facilities impact = facility_rating * internet_access
        df['facilities_impact'] = df['facility_rating_encoded'] * df['internet_access_encoded']
        
        # Age experience = age * study_hours
        df['age_study_interaction'] = df['age'] * df['study_hours']
        
        # Difficulty adjusted study = study_hours / (exam_difficulty_encoded + 1)
        df['difficulty_adjusted_study'] = df['study_hours'] / (df['exam_difficulty_encoded'] + 1)
        
        # Attendance sleep balance
        df['attendance_sleep_balance'] = df['class_attendance'] * df['sleep_hours'] / 100
        
        return df
    
    def _create_polynomial_features(self, df):
        """Create polynomial features for key numerical variables"""
        # Square and cube of study hours
        df['study_hours_squared'] = df['study_hours'] ** 2
        df['study_hours_cubed'] = df['study_hours'] ** 3
        
        # Square of attendance
        df['attendance_squared'] = df['class_attendance'] ** 2
        
        # Sleep hours squared
        df['sleep_hours_squared'] = df['sleep_hours'] ** 2
        
        # Log transformations (adding small constant to avoid log(0))
        df['log_study_hours'] = np.log1p(df['study_hours'])
        df['log_attendance'] = np.log1p(df['class_attendance'])
        
        return df
    
    def _create_bins(self, df):
        """Create binned categorical features"""
        # Age groups
        df['age_group'] = pd.cut(df['age'], bins=[16, 19, 21, 25], labels=['young', 'mid', 'senior'])
        df['age_group_encoded'] = LabelEncoder().fit_transform(df['age_group'].astype(str))
        
        # Study hours categories
        df['study_intensity'] = pd.cut(df['study_hours'], 
                                        bins=[-0.01, 2, 4, 6, 10], 
                                        labels=['low', 'medium', 'high', 'very_high'])
        df['study_intensity_encoded'] = LabelEncoder().fit_transform(df['study_intensity'].astype(str))
        
        # Attendance categories
        df['attendance_level'] = pd.cut(df['class_attendance'], 
                                         bins=[0, 50, 75, 90, 100], 
                                         labels=['poor', 'average', 'good', 'excellent'])
        df['attendance_level_encoded'] = LabelEncoder().fit_transform(df['attendance_level'].astype(str))
        
        return df
    
    def _create_statistical_features(self, df):
        """Create statistical aggregation features"""
        # Ratios
        df['study_per_age'] = df['study_hours'] / df['age']
        df['sleep_per_age'] = df['sleep_hours'] / df['age']
        
        # Normalized features
        df['study_hours_norm'] = df['study_hours'] / df['study_hours'].max()
        df['attendance_norm'] = df['class_attendance'] / 100
        
        return df
    
    def _create_advanced_features(self, df):
        """Create advanced interaction and complex features"""
        # Triple interactions
        df['study_attend_sleep'] = df['study_hours'] * df['class_attendance'] * df['sleep_hours'] / 1000
        df['study_sleep_quality'] = df['study_hours'] * df['sleep_hours'] * (df['sleep_quality_encoded'] + 1)
        
        # Weighted combinations
        df['weighted_study'] = df['study_hours'] * 0.4 + df['class_attendance'] / 100 * 0.6
        df['prep_score'] = (df['study_hours'] * 2 + df['class_attendance'] / 10 + df['sleep_hours']) / 4
        
        # Distance from mean (how far from average student)
        df['study_deviation'] = np.abs(df['study_hours'] - df['study_hours'].mean())
        df['attendance_deviation'] = np.abs(df['class_attendance'] - df['class_attendance'].mean())
        
        # Ratios of ratios
        df['study_to_sleep_ratio'] = df['study_hours'] / (df['sleep_hours'] + 0.1)
        df['efficiency_score'] = (df['study_hours'] * df['class_attendance']) / (df['age'] * 10)
        
        # Complex formula based on domain knowledge
        df['student_quality'] = (df['study_hours'] * 0.3 + 
                                  df['class_attendance'] / 100 * 0.3 + 
                                  df['sleep_hours'] / 10 * 0.2 + 
                                  (df['sleep_quality_encoded'] + 1) * 0.2)
        
        # Internet × facility interaction
        df['tech_advantage'] = df['internet_access_encoded'] * df['facility_rating_encoded']
        
        return df
    
    def _create_course_statistics(self, train, test=None):
        """Create per-course aggregated statistics"""
        if config.TARGET_COL in train.columns:
            # Calculate course-level statistics from training data
            course_stats = train.groupby('course')[config.TARGET_COL].agg(['mean', 'std', 'median']).reset_index()
            course_stats.columns = ['course', 'course_mean_score', 'course_std_score', 'course_median_score']
            
            # Merge back to train
            train = train.merge(course_stats, on='course', how='left')
            
            # Apply to test
            if test is not None:
                test = test.merge(course_stats, on='course', how='left')
                # Fill missing with global mean
                global_mean = train[config.TARGET_COL].mean()
                test['course_mean_score'].fillna(global_mean, inplace=True)
                test['course_std_score'].fillna(train[config.TARGET_COL].std(), inplace=True)
                test['course_median_score'].fillna(train[config.TARGET_COL].median(), inplace=True)
        else:
            # For test-only scenarios
            train['course_mean_score'] = 0
            train['course_std_score'] = 0
            train['course_median_score'] = 0
        
        return train, test
    
    def _create_rank_features(self, df):
        """Create rank-based features"""
        # Rank students by study hours
        df['study_rank'] = df['study_hours'].rank(pct=True)
        df['attendance_rank'] = df['class_attendance'].rank(pct=True)
        df['sleep_rank'] = df['sleep_hours'].rank(pct=True)
        
        # Combined rank
        df['overall_rank'] = (df['study_rank'] + df['attendance_rank'] + df['sleep_rank']) / 3
        
        return df

def create_target_encoding_features(train, test, target_col=config.TARGET_COL, n_folds=5):
    """
    Create target encoding for categorical features using K-Fold
    to prevent leakage
    """
    train_encoded = train.copy()
    test_encoded = test.copy()
    
    for col in config.CATEGORICAL_FEATURES:
        # Initialize encoded column
        train_encoded[f'{col}_target_mean'] = 0.0
        test_encoded[f'{col}_target_mean'] = 0.0
        
        # Calculate global mean for test set
        global_mean = train[target_col].mean()
        test_means = train.groupby(col)[target_col].mean()
        test_encoded[f'{col}_target_mean'] = test_encoded[col].map(test_means).fillna(global_mean)
        
        # K-Fold target encoding for train to prevent leakage
        if 'fold' in train_encoded.columns:
            for fold in range(n_folds):
                train_fold = train_encoded[train_encoded['fold'] != fold]
                val_fold_idx = train_encoded['fold'] == fold
                
                means = train_fold.groupby(col)[target_col].mean()
                train_encoded.loc[val_fold_idx, f'{col}_target_mean'] = \
                    train_encoded.loc[val_fold_idx, col].map(means).fillna(global_mean)
        else:
            # If no folds, use global encoding (less ideal)
            means = train.groupby(col)[target_col].mean()
            train_encoded[f'{col}_target_mean'] = train_encoded[col].map(means).fillna(global_mean)
    
    print("Target encoding features created")
    return train_encoded, test_encoded
