<?php
/**
 * This class creates a label element
 *
 * @author durso
 */
namespace library\dom\elements\components;
use library\dom\elements\components\intext;

class label extends intext{
    
    
    public function __construct($value) {
        parent::__construct("label",$value);
        
    }

}