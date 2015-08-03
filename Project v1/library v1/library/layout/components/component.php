<?php

/**
 * Description of component
 *
 * @author durso
 */
namespace library\layout\components;
use library\layout\elements\element;


abstract class component extends element{
    protected $elements = array();
    protected $root;
    
    public function addElement(element $element){
        $this->elements[] = $element; 
    }
    
    
}
