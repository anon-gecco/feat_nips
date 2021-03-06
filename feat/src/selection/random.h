/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef RANDOM_H
#define RANDOM_H

namespace FT{
    ////////////////////////////////////////////////////////////////////////////////// Declarations
    /*!
     * @class Random
     */
    struct Random : SelectionOperator
    {
        /** Random based selection and survival methods. */

        Random(bool surv){ name = "random"; survival = surv; elitism = true;};
        
        ~Random(){}
       
        vector<size_t> select(Population& pop, const MatrixXd& F, const Parameters& params);
        vector<size_t> survive(Population& pop, const MatrixXd& F, const Parameters& params);
        /// replaces worst individual in selected with best individual in Pop.
        void enforce_elite(Population& pop, vector<size_t>& selected);
        bool elitism;       //< whether or not to keep the best individual.

    };
    
    /////////////////////////////////////////////////////////////////////////////////// Definitions
    vector<size_t> Random::select(Population& pop, const MatrixXd& F, const Parameters& params)
    {
        /* Selects parents for making offspring.  
         *
         * @params pop: population of programs, all potential parents. 
         * @params F: n_samples x 2 * popsize matrix of program behaviors. 
         * @params params: parameters.
         *
         * @returns selected: vector of indices corresponding to offspring that are selected.
         *      
         */
      
        int P = pop.size(); // index P is where the offspring begin, and also the size of the pop
       
        vector<size_t> all_idx(pop.size());
        std::iota(all_idx.begin(), all_idx.end(), 0);

        cout << "selecting randoms\n";       
        vector<size_t> selected;
        for (unsigned i = 0; i < P; ++i)    
            selected.push_back(r.random_choice(all_idx));   // select randomly
       
        cout << "getting elite\n";       
        if (elitism)
            enforce_elite(pop, selected); 
        
        return selected;
    }

    vector<size_t> Random::survive(Population& pop, const MatrixXd& F, const Parameters& params)
    {
        /* Selects the offspring for survival. 
         *
         * @params pop: population of programs, parents + offspring.
         * @params F: n_samples x 2 * popsize matrix of program behaviors. 
         * @params params: parameters.
         *
         * @returns selected: vector of indices corresponding to offspring that are selected.
         *      
         */
      
        int P = F.cols()/2; // index P is where the offspring begin, and also the size of the pop
       
        vector<size_t> all_idx(pop.size());
        std::iota(all_idx.begin(), all_idx.end(), 0);

        cout << "selecting randoms\n";       
        vector<size_t> selected;
        for (unsigned i = 0; i < P; ++i)    
            selected.push_back(r.random_choice(all_idx));   // select randomly
        cout << "getting elite\n";       
        if (elitism)
            enforce_elite(pop, selected); 
        
        return selected;
    }
   
    void Random::enforce_elite(Population& pop, vector<size_t>& selected)
    {
        // find best and worst inds and if best is not in selected, replace worst with it
        size_t best_idx, worst_idx;
        double min_fit, max_fit;

        for (unsigned i = 0; i < pop.individuals.size(); ++i)
        {
            if (pop.individuals.at(i).fitness < min_fit || i == 0)
            {
                min_fit = pop.individuals.at(i).fitness;
                best_idx = i;
            }
        }    
        
        if (!in(selected, best_idx) )   // add best individual to selected
        {
            for (unsigned i = 0; i < selected.size(); ++i)
            {
                if (pop.individuals.at(selected.at(i)).fitness > max_fit || i == 0)
                {
                    max_fit = pop.individuals.at(selected.at(i)).fitness;
                    worst_idx = i;
                }
                
            }
            selected.at(worst_idx) = best_idx;
        }
        
    }

}
#endif
