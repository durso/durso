<?php

/**
 * Description of inline
 *
 * @author durso
 */
namespace library\dom\elements\components;
use library\dom\object;
use library\dom\elements\components\intext;

class title extends intext{

    
    public function __construct($value = false) {
        parent::__construct("title",$value);
        
    }
   


}
