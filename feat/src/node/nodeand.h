/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_AND
#define NODE_AND

#include "node.h"

namespace FT{
	class NodeAnd : public Node
    {
    	public:
    	
    		NodeAnd()
       		{
    			name = "and";
    			otype = 'b';
    			arity['f'] = 0;
    			arity['b'] = 2;
    			complexity = 2;
    		}
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(Data& data, Stacks& stack)
            {
                stack.b.push(stack.b.pop() && stack.b.pop());

            }

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack)
            {
                stack.bs.push("(" + stack.bs.pop() + " AND " + stack.bs.pop() + ")");
            }
        protected:
            virtual NodeAnd* clone_impl() const override { return new NodeAnd(*this); };  
            virtual NodeAnd* rnd_clone_impl() const override { return new NodeAnd(); };  
    };
}	

#endif
