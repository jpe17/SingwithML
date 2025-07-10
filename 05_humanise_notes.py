import csv
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from scipy import signal
from scipy.interpolate import interp1d
import json
import random
import math

@dataclass
class VocalNote:
    start_time: float
    duration: float
    midi_note: int
    velocity: float = 80
    pitch_bend: List[Tuple[float, float]] = None
    vibrato: Optional[Dict] = None
    breath_before: float = 0.0
    consonant_type: str = "none"

class AdvancedVocalHumanizer:
    def __init__(self):
        # Advanced parameters with musical intelligence
        self.vocal_range_comfort = {
            'soprano': (60, 81),    # C4 to A5
            'alto': (55, 74),       # G3 to D5
            'tenor': (48, 69),      # C3 to A4
            'bass': (40, 62)        # E2 to D4
        }
        
        # Psychoacoustic timing models
        self.perceptual_timing = {
            'rush_tendency': 0.008,     # Singers tend to rush fast passages
            'drag_tendency': 0.012,     # Drag on emotional peaks
            'syncopation_feel': 0.015,  # Natural syncopation
            'phrase_arch': 0.025        # Phrase temporal shaping
        }
        
        # Vocal production constraints
        self.vocal_physics = {
            'min_attack_time': 0.08,    # Minimum vocal onset
            'max_sustained': 8.0,       # Maximum breath support
            'consonant_duration': 0.05,  # Consonant articulation time
            'glottal_recovery': 0.1,    # Recovery between notes
            'vibrato_threshold': 0.4    # Minimum duration for vibrato
        }
        
        # Advanced musical pattern recognition
        self.pattern_weights = {
            'stepwise_motion': 1.0,
            'perfect_fourth': 0.8,
            'perfect_fifth': 0.7,
            'octave_leap': 0.6,
            'tritone': 0.3,
            'major_seventh': 0.2
        }
        
        # Neural network-inspired vocal modeling
        self.vocal_model_params = self._initialize_vocal_model()
    
    def _initialize_vocal_model(self) -> Dict:
        """Initialize AI-enhanced vocal behavior model"""
        return {
            'emotion_mapping': {
                'neutral': {'vibrato_rate': 5.5, 'timing_variance': 0.01},
                'expressive': {'vibrato_rate': 6.2, 'timing_variance': 0.02},
                'intense': {'vibrato_rate': 7.1, 'timing_variance': 0.025}
            },
            'formant_transitions': {
                'vowel_targets': [800, 1200, 2400],  # Simplified formant frequencies
                'transition_time': 0.15
            },
            'breath_model': {
                'capacity': 12.0,  # Seconds of comfortable singing
                'recovery_rate': 0.8,
                'phrase_planning': True
            }
        }
    
    def analyze_melodic_context(self, midi_data: List[Tuple[float, float, int]]) -> Dict:
        """Advanced melodic analysis using music theory"""
        if len(midi_data) < 3:
            return {'complexity': 'simple', 'key_signature': 'C', 'phrase_structure': []}
        
        notes = [note for _, _, note in midi_data]
        intervals = [notes[i+1] - notes[i] for i in range(len(notes)-1)]
        
        # Analyze interval patterns
        interval_analysis = {
            'stepwise_ratio': sum(1 for i in intervals if abs(i) <= 2) / len(intervals),
            'leap_ratio': sum(1 for i in intervals if abs(i) > 4) / len(intervals),
            'direction_changes': sum(1 for i in range(len(intervals)-1) 
                                   if intervals[i] * intervals[i+1] < 0),
            'avg_interval': np.mean([abs(i) for i in intervals]),
            'range_span': max(notes) - min(notes)
        }
        
        # Estimate key signature (simplified)
        pitch_classes = [note % 12 for note in notes]
        key_signature = self._estimate_key(pitch_classes)
        
        # Phrase boundary detection
        phrase_boundaries = self._detect_phrase_boundaries(midi_data)
        
        return {
            'complexity': 'complex' if interval_analysis['leap_ratio'] > 0.3 else 'moderate',
            'key_signature': key_signature,
            'phrase_structure': phrase_boundaries,
            'interval_analysis': interval_analysis,
            'tessitura': self._analyze_tessitura(notes)
        }
    
    def _estimate_key(self, pitch_classes: List[int]) -> str:
        """Simplified key estimation using pitch class distribution"""
        major_profiles = {
            'C': [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
            'G': [2.88, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29],
            'D': [2.29, 2.88, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66],
            'A': [3.66, 2.29, 2.88, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39],
            'E': [2.39, 3.66, 2.29, 2.88, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19],
            'F': [5.19, 2.39, 3.66, 2.29, 2.88, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52]
        }
        
        pc_histogram = [pitch_classes.count(i) for i in range(12)]
        correlations = {}
        
        for key, profile in major_profiles.items():
            correlation = np.corrcoef(pc_histogram, profile)[0, 1]
            correlations[key] = correlation if not np.isnan(correlation) else 0
        
        return max(correlations, key=correlations.get)
    
    def _detect_phrase_boundaries(self, midi_data: List[Tuple[float, float, int]]) -> List[int]:
        """Detect phrase boundaries using gap analysis and melodic contour"""
        boundaries = [0]  # Start of piece
        
        for i in range(1, len(midi_data)):
            current_start = midi_data[i][0]
            prev_end = midi_data[i-1][0] + midi_data[i-1][1]
            gap = current_start - prev_end
            
            # Large gap indicates phrase boundary
            if gap > 0.5:
                boundaries.append(i)
            
            # Melodic leap combined with rhythm change
            if i > 1:
                prev_interval = abs(midi_data[i-1][2] - midi_data[i-2][2])
                curr_interval = abs(midi_data[i][2] - midi_data[i-1][2])
                if prev_interval > 5 and curr_interval > 5:
                    boundaries.append(i)
        
        boundaries.append(len(midi_data))  # End of piece
        return boundaries
    
    def _analyze_tessitura(self, notes: List[int]) -> Dict:
        """Analyze vocal tessitura and comfort zone"""
        note_weights = {}
        for note in notes:
            note_weights[note] = note_weights.get(note, 0) + 1
        
        # Find the most comfortable range
        weighted_center = sum(note * weight for note, weight in note_weights.items()) / sum(note_weights.values())
        
        # Determine voice type based on tessitura
        voice_type = 'alto'  # Default
        for vtype, (low, high) in self.vocal_range_comfort.items():
            if low <= weighted_center <= high:
                voice_type = vtype
                break
        
        return {
            'center': weighted_center,
            'voice_type': voice_type,
            'comfort_score': self._calculate_comfort_score(notes, voice_type)
        }
    
    def _calculate_comfort_score(self, notes: List[int], voice_type: str) -> float:
        """Calculate how comfortable the melody is for the voice type"""
        comfort_range = self.vocal_range_comfort[voice_type]
        comfortable_notes = sum(1 for note in notes if comfort_range[0] <= note <= comfort_range[1])
        return comfortable_notes / len(notes)
    
    def apply_advanced_timing_model(self, midi_data: List[Tuple[float, float, int]], 
                                  musical_context: Dict) -> List[VocalNote]:
        """Apply sophisticated timing model based on musical context"""
        vocal_notes = []
        phrase_boundaries = musical_context['phrase_structure']
        
        for i, (start_time, duration, note) in enumerate(midi_data):
            # Determine phrase position
            phrase_position = self._get_phrase_position(i, phrase_boundaries)
            
            # Apply perceptual timing adjustments
            timing_adjustment = self._calculate_timing_adjustment(
                i, midi_data, phrase_position, musical_context
            )
            
            # Apply duration modeling
            duration_adjustment = self._calculate_duration_adjustment(
                duration, note, phrase_position, musical_context
            )
            
            # Create advanced vocal note
            vocal_note = VocalNote(
                start_time=start_time + timing_adjustment,
                duration=duration * duration_adjustment,
                midi_note=note,
                velocity=self._calculate_dynamic_velocity(note, phrase_position),
                pitch_bend=self._generate_pitch_bend(note, duration),
                vibrato=self._generate_vibrato(note, duration),
                breath_before=self._calculate_breath_requirement(i, midi_data, phrase_boundaries),
                consonant_type=self._estimate_consonant_type(i, midi_data)
            )
            
            vocal_notes.append(vocal_note)
        
        return vocal_notes
    
    def _get_phrase_position(self, note_index: int, phrase_boundaries: List[int]) -> Dict:
        """Determine position within phrase structure"""
        for i in range(len(phrase_boundaries) - 1):
            if phrase_boundaries[i] <= note_index < phrase_boundaries[i + 1]:
                phrase_length = phrase_boundaries[i + 1] - phrase_boundaries[i]
                position_in_phrase = note_index - phrase_boundaries[i]
                return {
                    'phrase_number': i,
                    'position_ratio': position_in_phrase / phrase_length,
                    'is_phrase_start': position_in_phrase == 0,
                    'is_phrase_end': position_in_phrase == phrase_length - 1,
                    'phrase_length': phrase_length
                }
        return {'phrase_number': 0, 'position_ratio': 0.5, 'is_phrase_start': False, 'is_phrase_end': False, 'phrase_length': 1}
    
    def _calculate_timing_adjustment(self, note_index: int, midi_data: List[Tuple[float, float, int]], 
                                   phrase_position: Dict, musical_context: Dict) -> float:
        """Calculate sophisticated timing adjustments"""
        base_variance = random.uniform(-0.01, 0.01)  # Basic humanization
        
        # Phrase-based timing
        if phrase_position['is_phrase_start']:
            base_variance += random.uniform(-0.005, 0.02)  # Slight anticipation or delay
        elif phrase_position['is_phrase_end']:
            base_variance += random.uniform(0.01, 0.03)  # Natural ritardando
        
        # Musical complexity adjustment
        if musical_context['complexity'] == 'complex':
            base_variance *= 1.5
        
        # Interval-based timing
        if note_index > 0:
            prev_note = midi_data[note_index - 1][2]
            current_note = midi_data[note_index][2]
            interval = abs(current_note - prev_note)
            
            if interval > 7:  # Large leap
                base_variance += 0.015  # Slight hesitation before large leaps
            elif interval <= 1:  # Stepwise motion
                base_variance -= 0.005  # Smoother connection
        
        return base_variance
    
    def _calculate_duration_adjustment(self, duration: float, note: int, 
                                     phrase_position: Dict, musical_context: Dict) -> float:
        """Calculate intelligent duration adjustments"""
        base_factor = 1.0
        
        # Ensure minimum vocal duration
        if duration < self.vocal_physics['min_attack_time']:
            base_factor = self.vocal_physics['min_attack_time'] / duration
        
        # High note emphasis
        if note > 72:  # Above C5
            base_factor *= 1.1
        
        # Phrase ending elongation
        if phrase_position['is_phrase_end']:
            base_factor *= 1.2
        
        # Prevent overly long notes
        if duration * base_factor > self.vocal_physics['max_sustained']:
            base_factor = self.vocal_physics['max_sustained'] / duration
        
        return base_factor
    
    def _calculate_dynamic_velocity(self, note: int, phrase_position: Dict) -> float:
        """Calculate realistic velocity based on musical context"""
        base_velocity = 80
        
        # High notes get more emphasis
        if note > 72:
            base_velocity += 15
        elif note < 55:
            base_velocity -= 10
        
        # Phrase shaping
        phrase_ratio = phrase_position['position_ratio']
        if phrase_ratio < 0.3:  # Beginning of phrase
            base_velocity -= 5
        elif phrase_ratio > 0.7:  # End of phrase
            base_velocity += 8
        
        # Add small random variation
        base_velocity += random.uniform(-5, 5)
        
        return max(40, min(127, base_velocity))
    
    def _generate_pitch_bend(self, note: int, duration: float) -> List[Tuple[float, float]]:
        """Generate realistic pitch bend curves"""
        if duration < 0.3:
            return []
        
        pitch_bend_points = []
        
        # Subtle pitch bend at note onset (vocal attack characteristics)
        pitch_bend_points.append((0.0, random.uniform(-10, 10)))
        
        # Stabilization
        pitch_bend_points.append((0.1, random.uniform(-2, 2)))
        
        # Possible vibrato-like modulation for longer notes
        if duration > 0.8:
            num_oscillations = int(duration * 5.5)  # ~5.5 Hz vibrato
            for i in range(num_oscillations):
                time_point = 0.3 + (i / num_oscillations) * (duration - 0.3)
                bend_amount = random.uniform(-8, 8) * math.sin(2 * math.pi * 5.5 * time_point)
                pitch_bend_points.append((time_point, bend_amount))
        
        return pitch_bend_points
    
    def _generate_vibrato(self, note: int, duration: float) -> Optional[Dict]:
        """Generate realistic vibrato parameters"""
        if duration < self.vocal_physics['vibrato_threshold']:
            return None
        
        # Vibrato is more common on higher notes and longer durations
        vibrato_probability = min(0.8, (duration - 0.4) * 0.5 + (note - 60) * 0.02)
        
        if random.random() < vibrato_probability:
            return {
                'rate': random.uniform(5.0, 6.5),  # Hz
                'depth': random.uniform(0.3, 0.8),  # Semitones
                'onset_delay': random.uniform(0.2, 0.4),  # Seconds
                'fade_in_time': random.uniform(0.1, 0.3)
            }
        
        return None
    
    def _calculate_breath_requirement(self, note_index: int, midi_data: List[Tuple[float, float, int]], 
                                    phrase_boundaries: List[int]) -> float:
        """Calculate breath requirements using physiological modeling"""
        if note_index == 0:
            return 0.1  # Initial breath
        
        # Check if at phrase boundary
        if note_index in phrase_boundaries:
            return random.uniform(0.15, 0.3)  # Phrase breath
        
        # Check accumulated breath usage
        breath_usage = 0
        start_index = max(0, note_index - 10)  # Look back 10 notes
        
        for i in range(start_index, note_index):
            duration = midi_data[i][1]
            note = midi_data[i][2]
            
            # Higher notes and longer durations use more breath
            breath_cost = duration * (1.0 + (note - 60) * 0.02)
            breath_usage += breath_cost
        
        # If breath usage is high, insert a small breath
        if breath_usage > 6.0:  # Half breath capacity
            return random.uniform(0.05, 0.15)
        
        return 0.0
    
    def _estimate_consonant_type(self, note_index: int, midi_data: List[Tuple[float, float, int]]) -> str:
        """Estimate consonant type based on melodic patterns"""
        if note_index == 0:
            return "hard_attack"
        
        prev_note = midi_data[note_index - 1][2]
        current_note = midi_data[note_index][2]
        interval = abs(current_note - prev_note)
        
        if interval == 0:
            return "legato"
        elif interval <= 2:
            return "smooth"
        elif interval <= 5:
            return "moderate"
        else:
            return "articulated"
    
    def export_advanced_midi(self, vocal_notes: List[VocalNote], output_file: str):
        """Export with advanced features"""
        # Standard CSV export
        with open(output_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['start_time_sec', 'duration_sec', 'midi_note', 'velocity', 
                           'breath_before', 'consonant_type', 'has_vibrato'])
            
            for note in vocal_notes:
                writer.writerow([
                    f"{note.start_time:.6f}",
                    f"{note.duration:.6f}",
                    note.midi_note,
                    f"{note.velocity:.1f}",
                    f"{note.breath_before:.3f}",
                    note.consonant_type,
                    note.vibrato is not None
                ])
        
        # Advanced JSON export with full data
        advanced_data = {
            'metadata': {
                'version': '2.0',
                'features': ['advanced_timing', 'breath_modeling', 'pitch_bend', 'vibrato', 'consonant_analysis']
            },
            'notes': []
        }
        
        for note in vocal_notes:
            note_data = {
                'start_time': note.start_time,
                'duration': note.duration,
                'midi_note': note.midi_note,
                'velocity': note.velocity,
                'breath_before': note.breath_before,
                'consonant_type': note.consonant_type,
                'pitch_bend': note.pitch_bend or [],
                'vibrato': note.vibrato
            }
            advanced_data['notes'].append(note_data)
        
        json_file = output_file.replace('.csv', '_advanced.json')
        with open(json_file, 'w') as f:
            json.dump(advanced_data, f, indent=2)
        
        print(f"Advanced MIDI data exported to {output_file} and {json_file}")
    
    def humanize_advanced(self, input_file: str, output_file: str):
        """Main advanced humanization process"""
        print("🎵 Loading and analyzing MIDI data...")
        
        # Load data
        midi_data = []
        with open(input_file, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                start_time = float(row['start_time_sec'])
                duration = float(row['duration_sec'])
                midi_note = int(row['midi_note'])
                midi_data.append((start_time, duration, midi_note))
        
        # Advanced musical analysis
        print("🧠 Performing advanced musical analysis...")
        musical_context = self.analyze_melodic_context(midi_data)
        
        print(f"   Key signature: {musical_context['key_signature']}")
        print(f"   Complexity: {musical_context['complexity']}")
        print(f"   Voice type: {musical_context['tessitura']['voice_type']}")
        print(f"   Comfort score: {musical_context['tessitura']['comfort_score']:.2f}")
        
        # Apply advanced humanization
        print("🎭 Applying advanced vocal humanization...")
        vocal_notes = self.apply_advanced_timing_model(midi_data, musical_context)
        
        # Export results
        print("💾 Exporting results...")
        self.export_advanced_midi(vocal_notes, output_file)
        
        # Performance analysis
        self._print_advanced_analysis(midi_data, vocal_notes, musical_context)
    
    def _print_advanced_analysis(self, original: List[Tuple[float, float, int]], 
                               processed: List[VocalNote], musical_context: Dict):
        """Print comprehensive analysis"""
        print("\n🎯 ADVANCED HUMANIZATION ANALYSIS")
        print("="*50)
        
        print(f"Musical Context:")
        print(f"  • Key: {musical_context['key_signature']}")
        print(f"  • Phrases: {len(musical_context['phrase_structure'])-1}")
        print(f"  • Stepwise motion: {musical_context['interval_analysis']['stepwise_ratio']:.1%}")
        print(f"  • Voice type: {musical_context['tessitura']['voice_type']}")
        
        print(f"\nVocal Enhancements:")
        vibrato_count = sum(1 for note in processed if note.vibrato is not None)
        breath_count = sum(1 for note in processed if note.breath_before > 0.05)
        
        print(f"  • Notes with vibrato: {vibrato_count}/{len(processed)} ({vibrato_count/len(processed):.1%})")
        print(f"  • Breath points added: {breath_count}")
        print(f"  • Avg timing variance: {np.mean([abs(p.start_time - o[0]) for o, p in zip(original, processed)]):.3f}s")
        
        print(f"\nSample Transformations:")
        for i in range(min(3, len(original))):
            orig = original[i]
            proc = processed[i]
            print(f"  Note {i+1}: {orig[0]:.3f}s → {proc.start_time:.3f}s "
                  f"({proc.start_time-orig[0]:+.3f}s)")
            print(f"           {orig[1]:.3f}s → {proc.duration:.3f}s "
                  f"({proc.duration-orig[1]:+.3f}s)")
            if proc.vibrato:
                print(f"           + Vibrato: {proc.vibrato['rate']:.1f}Hz")
            if proc.breath_before > 0.01:
                print(f"           + Breath: {proc.breath_before:.3f}s")

# Demo usage
if __name__ == "__main__":
    # Process with advanced humanizer
    humanizer = AdvancedVocalHumanizer()
    humanizer.humanize_advanced('04_extracted_notes/voice_005000_carpentersclosetoyou_0000.csv', '05_humanise_notes/voice_005000_carpentersclosetoyou_0000.csv')
    
    print("\n✨ Advanced processing complete!")
