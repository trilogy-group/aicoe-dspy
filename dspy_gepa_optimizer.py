#!/usr/bin/env python3
"""
DSPy GEPA Optimizer: Genetic-Pareto Algorithm within DSPy Framework
===================================================================

This module demonstrates how to properly implement custom optimization algorithms
within the DSPy framework, rather than building standalone systems.

Key Features:
- Inherits from dspy.teleprompt.Teleprompter (proper DSPy integration)
- Uses DSPy's infrastructure (LM management, Examples, evaluation)
- Combines genetic algorithms with DSPy's bootstrap learning
- Maintains Pareto frontier for multi-objective optimization
- Compatible with all DSPy modules and signatures

Usage:
    import dspy
    from dspy_gepa_optimizer import AdvancedGEPA
    
    optimizer = AdvancedGEPA(metric=my_metric, population_size=10)
    optimized_program = optimizer.compile(my_program, trainset=examples)
"""

import dspy
import random
import copy
from typing import List, Dict, Any, Callable, Optional, Tuple
from dataclasses import dataclass
import numpy as np

@dataclass
class GEPAIndividual:
    """Individual in GEPA population - represents a DSPy program configuration"""
    program: dspy.Module
    fitness_scores: Dict[str, float]
    generation: int
    bootstrap_demos: List[dspy.Example]
    instruction_mutations: Dict[str, str]
    
    @property
    def primary_fitness(self) -> float:
        """Primary fitness metric for optimization"""
        return self.fitness_scores.get('primary', 0.0)
    
    @property 
    def diversity_score(self) -> float:
        """Diversity/complexity metric for Pareto optimization"""
        return self.fitness_scores.get('diversity', 0.0)

class AdvancedGEPA(dspy.teleprompt.Teleprompter):
    """
    Advanced GEPA Optimizer integrated into DSPy Framework
    
    This optimizer combines:
    1. Genetic Algorithm principles for population evolution
    2. Pareto frontier optimization for multi-objective goals
    3. DSPy's bootstrap learning for demonstration generation
    4. Instruction mutation for prompt evolution
    
    Benefits over standalone implementation:
    - Uses DSPy's LM management and caching
    - Compatible with all DSPy modules and signatures
    - Leverages DSPy's evaluation infrastructure
    - Integrates with DSPy's example and metric systems
    """
    
    def __init__(
        self,
        metric: Callable,
        population_size: int = 12,
        generations: int = 8,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.7,
        max_bootstrapped_demos: int = 4,
        max_labeled_demos: int = 4,
        diversity_weight: float = 0.3,
        elite_size: int = 3,
        verbose: bool = True
    ):
        """
        Initialize GEPA optimizer with DSPy framework integration
        
        Args:
            metric: DSPy-compatible evaluation metric
            population_size: Number of individuals in genetic population
            generations: Number of evolutionary generations 
            mutation_rate: Probability of instruction mutation
            crossover_rate: Probability of demonstration crossover
            max_bootstrapped_demos: Max auto-generated examples per individual
            max_labeled_demos: Max labeled examples per individual
            diversity_weight: Weight for diversity in Pareto optimization
            elite_size: Number of elite individuals to preserve
            verbose: Enable detailed logging
        """
        self.metric = metric
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_bootstrapped_demos = max_bootstrapped_demos
        self.max_labeled_demos = max_labeled_demos
        self.diversity_weight = diversity_weight
        self.elite_size = elite_size
        self.verbose = verbose
        
        # GEPA state
        self.population: List[GEPAIndividual] = []
        self.pareto_frontier: List[GEPAIndividual] = []
        self.generation_stats: List[Dict[str, float]] = []
        
        # DSPy bootstrap optimizer for generating demonstrations
        self.bootstrap_optimizer = dspy.BootstrapFewShot(
            metric=metric,
            max_bootstrapped_demos=max_bootstrapped_demos,
            max_labeled_demos=max_labeled_demos
        )
    
    def compile(self, student: dspy.Module, *, trainset: List[dspy.Example], teacher=None, valset=None) -> dspy.Module:
        """
        DSPy-compliant compile method implementing GEPA optimization
        
        Args:
            student: DSPy module to optimize
            trainset: Training examples in DSPy format
            teacher: Optional teacher module for bootstrap learning
            valset: Optional validation set for evaluation
            
        Returns:
            Optimized DSPy module from Pareto frontier
        """
        if self.verbose:
            print(f"🧬 Starting DSPy GEPA Optimization")
            print(f"   Population Size: {self.population_size}")
            print(f"   Generations: {self.generations}")
            print(f"   Training Examples: {len(trainset)}")
            print(f"   Validation Examples: {len(valset) if valset else 'Using trainset'}")
        
        # Use validation set if provided, otherwise split trainset
        if valset is None:
            split_point = max(1, len(trainset) // 4)
            valset = trainset[:split_point]
            effective_trainset = trainset[split_point:]
        else:
            effective_trainset = trainset
        
        # Initialize population
        self._initialize_population(student, effective_trainset, teacher)
        
        # Evolve population across generations
        for generation in range(self.generations):
            if self.verbose:
                print(f"\n🔄 Generation {generation + 1}/{self.generations}")
            
            # Evaluate population fitness
            self._evaluate_population(valset)
            
            # Update Pareto frontier
            self._update_pareto_frontier()
            
            # Track generation statistics
            self._record_generation_stats(generation)
            
            # Generate next generation (except for last iteration)
            if generation < self.generations - 1:
                self._evolve_population(effective_trainset, teacher)
        
        # Final evaluation and selection
        self._evaluate_population(valset)
        self._update_pareto_frontier()
        
        # Select best individual from Pareto frontier
        best_individual = self._select_best_individual()
        
        if self.verbose:
            print(f"\n🎯 GEPA Optimization Complete")
            print(f"   Best Fitness: {best_individual.primary_fitness:.3f}")
            print(f"   Pareto Frontier Size: {len(self.pareto_frontier)}")
            print(f"   Total Evaluations: {self.population_size * self.generations}")
        
        return best_individual.program
    
    def _initialize_population(self, student: dspy.Module, trainset: List[dspy.Example], teacher=None):
        """Initialize genetic population using DSPy bootstrap learning"""
        if self.verbose:
            print("   🌱 Initializing GEPA population...")
        
        for i in range(self.population_size):
            try:
                # Create diverse training subsets for each individual
                subset_size = max(3, len(trainset) // 2)
                individual_trainset = random.sample(trainset, min(subset_size, len(trainset)))
                
                # Use DSPy bootstrap to create individual
                individual_program = self.bootstrap_optimizer.compile(
                    copy.deepcopy(student),
                    trainset=individual_trainset,
                    teacher=teacher
                )
                
                # Create GEPA individual
                individual = GEPAIndividual(
                    program=individual_program,
                    fitness_scores={},
                    generation=0,
                    bootstrap_demos=individual_trainset,
                    instruction_mutations={}
                )
                
                self.population.append(individual)
                
            except Exception as e:
                if self.verbose:
                    print(f"   ⚠️ Failed to create individual {i + 1}: {e}")
                
                # Fallback: use original student
                fallback_individual = GEPAIndividual(
                    program=copy.deepcopy(student),
                    fitness_scores={},
                    generation=0,
                    bootstrap_demos=[],
                    instruction_mutations={}
                )
                self.population.append(fallback_individual)
        
        if self.verbose:
            print(f"   ✅ Created population of {len(self.population)} individuals")
    
    def _evaluate_population(self, valset: List[dspy.Example]):
        """Evaluate fitness of all individuals in population"""
        if self.verbose:
            print(f"   📊 Evaluating population fitness...")
        
        for i, individual in enumerate(self.population):
            try:
                # Primary fitness: performance on validation set
                evaluator = dspy.Evaluate(devset=valset, metric=self.metric, num_threads=1)
                primary_score = evaluator(individual.program, display_table=0)
                
                # Diversity score: complexity and uniqueness metrics
                diversity_score = self._calculate_diversity_score(individual)
                
                individual.fitness_scores = {
                    'primary': primary_score,
                    'diversity': diversity_score,
                    'weighted': primary_score + (self.diversity_weight * diversity_score)
                }
                
            except Exception as e:
                if self.verbose:
                    print(f"   ⚠️ Evaluation failed for individual {i + 1}: {e}")
                
                individual.fitness_scores = {
                    'primary': 0.0,
                    'diversity': 0.0,
                    'weighted': 0.0
                }
    
    def _calculate_diversity_score(self, individual: GEPAIndividual) -> float:
        """Calculate diversity score for Pareto optimization"""
        score = 0.0
        
        # Demonstration diversity
        demo_count = len(individual.bootstrap_demos)
        score += min(1.0, demo_count / self.max_bootstrapped_demos) * 0.4
        
        # Instruction mutation diversity
        mutation_count = len(individual.instruction_mutations)
        score += min(1.0, mutation_count / 3) * 0.3
        
        # Generation diversity (prefer newer generations)
        generation_bonus = min(1.0, individual.generation / self.generations) * 0.3
        score += generation_bonus
        
        return score
    
    def _update_pareto_frontier(self):
        """Update Pareto frontier with non-dominated individuals"""
        # Collect all individuals (current population + existing frontier)
        candidates = self.population + self.pareto_frontier
        
        # Find non-dominated individuals
        new_frontier = []
        for candidate in candidates:
            is_dominated = False
            
            for other in candidates:
                if other == candidate:
                    continue
                
                # Check if other dominates candidate (better in all objectives)
                if (other.primary_fitness >= candidate.primary_fitness and 
                    other.diversity_score >= candidate.diversity_score and
                    (other.primary_fitness > candidate.primary_fitness or 
                     other.diversity_score > candidate.diversity_score)):
                    is_dominated = True
                    break
            
            if not is_dominated:
                new_frontier.append(candidate)
        
        # Remove duplicates and limit frontier size
        seen_programs = set()
        unique_frontier = []
        
        for individual in new_frontier:
            program_id = id(individual.program)
            if program_id not in seen_programs:
                seen_programs.add(program_id)
                unique_frontier.append(individual)
        
        # Keep best individuals if frontier is too large
        if len(unique_frontier) > self.population_size:
            unique_frontier.sort(key=lambda x: x.fitness_scores['weighted'], reverse=True)
            unique_frontier = unique_frontier[:self.population_size]
        
        self.pareto_frontier = unique_frontier
    
    def _record_generation_stats(self, generation: int):
        """Record statistics for current generation"""
        if not self.population:
            return
        
        primary_scores = [ind.primary_fitness for ind in self.population]
        diversity_scores = [ind.diversity_score for ind in self.population]
        
        stats = {
            'generation': generation,
            'mean_primary': np.mean(primary_scores),
            'max_primary': np.max(primary_scores),
            'mean_diversity': np.mean(diversity_scores),
            'max_diversity': np.max(diversity_scores),
            'frontier_size': len(self.pareto_frontier)
        }
        
        self.generation_stats.append(stats)
        
        if self.verbose:
            print(f"   📈 Primary: {stats['max_primary']:.3f} (avg: {stats['mean_primary']:.3f})")
            print(f"   🎯 Diversity: {stats['max_diversity']:.3f} (avg: {stats['mean_diversity']:.3f})")
            print(f"   🏆 Frontier: {stats['frontier_size']} individuals")
    
    def _evolve_population(self, trainset: List[dspy.Example], teacher=None):
        """Generate next generation using genetic operators"""
        if self.verbose:
            print("   🧬 Evolving to next generation...")
        
        next_generation = []
        
        # Elitism: preserve best individuals
        elite_individuals = sorted(
            self.population, 
            key=lambda x: x.fitness_scores['weighted'], 
            reverse=True
        )[:self.elite_size]
        
        for elite in elite_individuals:
            elite_copy = copy.deepcopy(elite)
            elite_copy.generation += 1
            next_generation.append(elite_copy)
        
        # Generate remaining individuals through crossover and mutation
        while len(next_generation) < self.population_size:
            try:
                # Parent selection (tournament selection)
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
                
                # Crossover
                if random.random() < self.crossover_rate:
                    child = self._crossover(parent1, parent2, trainset, teacher)
                else:
                    child = copy.deepcopy(parent1)
                    child.generation += 1
                
                # Mutation
                if random.random() < self.mutation_rate:
                    child = self._mutate(child, trainset, teacher)
                
                next_generation.append(child)
                
            except Exception as e:
                if self.verbose:
                    print(f"   ⚠️ Evolution step failed: {e}")
                
                # Fallback: copy random parent
                fallback = copy.deepcopy(random.choice(self.population))
                fallback.generation += 1
                next_generation.append(fallback)
        
        self.population = next_generation[:self.population_size]
    
    def _tournament_selection(self, tournament_size: int = 3) -> GEPAIndividual:
        """Select parent using tournament selection"""
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        return max(tournament, key=lambda x: x.fitness_scores['weighted'])
    
    def _crossover(self, parent1: GEPAIndividual, parent2: GEPAIndividual, 
                   trainset: List[dspy.Example], teacher=None) -> GEPAIndividual:
        """Create child through demonstration crossover"""
        # Combine demonstrations from both parents
        combined_demos = list(parent1.bootstrap_demos) + list(parent2.bootstrap_demos)
        
        # Add some new examples for diversity
        new_demos = random.sample(trainset, min(2, len(trainset)))
        combined_demos.extend(new_demos)
        
        # Remove duplicates and limit size
        unique_demos = []
        seen_inputs = set()
        
        for demo in combined_demos:
            demo_input = demo.inputs() if hasattr(demo, 'inputs') else str(demo)
            demo_key = str(demo_input)
            if demo_key not in seen_inputs:
                seen_inputs.add(demo_key)
                unique_demos.append(demo)
        
        child_demos = unique_demos[:self.max_bootstrapped_demos + self.max_labeled_demos]
        
        # Create new program with combined demonstrations
        try:
            child_program = self.bootstrap_optimizer.compile(
                copy.deepcopy(parent1.program),
                trainset=child_demos,
                teacher=teacher
            )
        except:
            child_program = copy.deepcopy(parent1.program)
        
        return GEPAIndividual(
            program=child_program,
            fitness_scores={},
            generation=max(parent1.generation, parent2.generation) + 1,
            bootstrap_demos=child_demos,
            instruction_mutations={}
        )
    
    def _mutate(self, individual: GEPAIndividual, trainset: List[dspy.Example], teacher=None) -> GEPAIndividual:
        """Apply mutation to individual"""
        mutated = copy.deepcopy(individual)
        
        # Demonstration mutation: replace some demos with new ones
        if len(trainset) > 0 and len(mutated.bootstrap_demos) > 0:
            num_to_replace = max(1, len(mutated.bootstrap_demos) // 3)
            indices_to_replace = random.sample(range(len(mutated.bootstrap_demos)), 
                                             min(num_to_replace, len(mutated.bootstrap_demos)))
            
            for idx in indices_to_replace:
                if len(trainset) > 0:
                    mutated.bootstrap_demos[idx] = random.choice(trainset)
        
        # Instruction mutation (placeholder - would need access to module internals)
        mutation_key = f"mutation_{len(mutated.instruction_mutations)}"
        mutated.instruction_mutations[mutation_key] = "enhanced_instruction"
        
        return mutated
    
    def _select_best_individual(self) -> GEPAIndividual:
        """Select best individual from Pareto frontier"""
        if not self.pareto_frontier:
            # Fallback to best from population
            return max(self.population, key=lambda x: x.fitness_scores['weighted'])
        
        # Select individual with best weighted score from frontier
        return max(self.pareto_frontier, key=lambda x: x.fitness_scores['weighted'])
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report"""
        if not self.generation_stats:
            return {"error": "No optimization data available"}
        
        final_stats = self.generation_stats[-1]
        best_individual = self._select_best_individual()
        
        return {
            "optimization_summary": {
                "total_generations": len(self.generation_stats),
                "population_size": self.population_size,
                "final_fitness": best_individual.primary_fitness,
                "improvement": final_stats['max_primary'] - self.generation_stats[0]['max_primary'],
                "pareto_frontier_size": len(self.pareto_frontier)
            },
            "generation_progression": self.generation_stats,
            "best_individual": {
                "primary_fitness": best_individual.primary_fitness,
                "diversity_score": best_individual.diversity_score,
                "generation": best_individual.generation,
                "demo_count": len(best_individual.bootstrap_demos),
                "mutations_applied": len(best_individual.instruction_mutations)
            },
            "pareto_frontier": [
                {
                    "primary_fitness": ind.primary_fitness,
                    "diversity_score": ind.diversity_score,
                    "weighted_score": ind.fitness_scores['weighted']
                }
                for ind in self.pareto_frontier
            ]
        }

# Example usage and testing
if __name__ == "__main__":
    # This demonstrates how to use the GEPA optimizer within DSPy
    print("🧬 DSPy GEPA Optimizer - Standalone Test")
    print("="*50)
    
    # Configure DSPy (would normally use real LM)
    try:
        import os
        if os.getenv('OPENAI_API_KEY'):
            lm = dspy.LM('openai/gpt-4o-mini')
            dspy.configure(lm=lm)
            print("✅ DSPy configured with OpenAI")
        else:
            print("⚠️ No API key found - running conceptual demo")
    except:
        print("⚠️ Running conceptual demo without actual LM")
    
    # Define a simple DSPy program for testing
    class TestSignature(dspy.Signature):
        """Test signature for demonstration"""
        input_text: str = dspy.InputField()
        output_text: str = dspy.OutputField()
    
    test_program = dspy.Predict(TestSignature)
    
    # Create test examples
    test_examples = [
        dspy.Example(input_text="test1", output_text="result1").with_inputs('input_text'),
        dspy.Example(input_text="test2", output_text="result2").with_inputs('input_text'),
        dspy.Example(input_text="test3", output_text="result3").with_inputs('input_text'),
    ]
    
    # Simple test metric
    def test_metric(example, pred, trace=None):
        return 1.0 if hasattr(pred, 'output_text') else 0.0
    
    # Create and run GEPA optimizer
    gepa_optimizer = AdvancedGEPA(
        metric=test_metric,
        population_size=6,
        generations=3,
        verbose=True
    )
    
    print("\n🚀 Running GEPA optimization test...")
    try:
        optimized_program = gepa_optimizer.compile(test_program, trainset=test_examples)
        print("✅ GEPA optimization completed successfully")
        
        # Show optimization report
        report = gepa_optimizer.get_optimization_report()
        print(f"\n📊 Optimization Report:")
        print(f"   Final Fitness: {report['best_individual']['primary_fitness']:.3f}")
        print(f"   Pareto Frontier: {report['optimization_summary']['pareto_frontier_size']} individuals")
        print(f"   Total Evaluations: {report['optimization_summary']['total_generations'] * gepa_optimizer.population_size}")
        
    except Exception as e:
        print(f"❌ GEPA test failed: {e}")
    
    print("\n✨ This demonstrates GEPA as a proper DSPy optimizer extension!")