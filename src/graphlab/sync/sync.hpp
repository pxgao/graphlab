/**  
 * Copyright (c) 2009 Carnegie Mellon University. 
 *     All rights reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing,
 *  software distributed under the License is distributed on an "AS
 *  IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 *  express or implied.  See the License for the specific language
 *  governing permissions and limitations under the License.
 *
 * For more about this software visit:
 *
 *      http://www.graphlab.ml.cmu.edu
 *
 */

/**
 * Also contains code that is Copyright 2011 Yahoo! Inc.  All rights
 * reserved.  
 *
 * Contributed under the iCLA for:
 *    Joseph Gonzalez (jegonzal@yahoo-inc.com) 
 *
 */


#ifndef GRAPHLAB_SYNC_HPP
#define	GRAPHLAB_SYNC_HPP



#include <graphlab/logger/logger.hpp>
#include <graphlab/sync/isync.hpp>



namespace graphlab {



  namespace sync_defaults {

    template<typename VertexData, typename Accum>
    void fold(const VertexData& vdata, Accum& accum) {
      accum += Accum(vdata);
    } // end of default_apply

    template<typename T, typename Accum>
    void apply(T& lvalue, const Accum& accum) {
      lvalue = T(accum);
    } // end of default_apply
  }; // end of sync_defaults namespace


  template<typename Graph, typename T, typename Accum >
  class fold_sync : public isync<Graph> {
  public:


    typedef isync<Graph> isync_type;
    typedef typename Graph::vertex_data_type vertex_data_type;
    //! The target should supply = operation
    typedef glshared<T> glshared_type;
    typedef T           contained_type;
    typedef Accum       accumulator_type;

    

    /** 
     * The map function takes a vertex data and extracts the relevant
     * information into the result object.
     */
    typedef void(*fold_function_type)(const vertex_data_type& vertex, 
                                      Accum& result);

    /**
     *  The apply function manipulates the partial sum and assigns it
     *  to the target gl shared object
     */
    typedef void(*apply_function_type)(T& lvalue, const Accum& accum);




  private:
    glshared_type& target;
    accumulator_type zero;
    accumulator_type acc;

    fold_function_type fold_function;
    apply_function_type apply_function;

  public:

    fold_sync(const fold_sync& other) : 
      target(other.target), zero(other.zero), acc(other.acc),
      fold_function(other.fold_function), 
      apply_function(other.apply_function) { 
      ASSERT_TRUE(fold_function != NULL);
      ASSERT_TRUE(apply_function != NULL);
    }

    fold_sync(glshared_type& target,
              const accumulator_type& zero = accumulator_type(0),
              fold_function_type fold_function = 
              (sync_defaults::fold<vertex_data_type, Accum>),
              apply_function_type apply_function = 
              (sync_defaults::apply<T, Accum>) ) :
      target(target), zero(zero), acc(zero),
      fold_function(fold_function), apply_function(apply_function) { 
      ASSERT_TRUE(fold_function != NULL);
      ASSERT_TRUE(apply_function != NULL);
    }


    
    isync_type* clone() const { return new fold_sync(*this); }
    void clear() { acc = zero; }
    void operator+=(const vertex_data_type& vdata) { 
      fold_function(vdata, acc); 
    }
    void operator+=(const isync_type& iother) {
      const fold_sync& other = 
        *dynamic_cast<const fold_sync*>(&iother);
      acc += other.acc;
    }
    void apply() { 
      target.apply(apply_function, acc); 
    }
  }; // end of fold_sync
    

  
}; // end of Namespace graphlab


#endif