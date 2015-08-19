<?php

/**
 * Description of component
 *
 * @author durso
 */
namespace library\dom\structures;
use library\dom\object;
use library\dom\elements\components\elementFactory;
use library\dom\elements\elementCollection;


abstract class components{
    protected $root;
    protected $components = array();

    public function __construct($tag){
        $this->root = elementFactory::createByTag($tag);
        $this->tracker($this->root);
    }
    public function addComponent(object $component){
        $this->root->addComponent($component);
        $this->tracker($component);
    }
    protected function tracker(object $component){
        if($component->hasTag()){
            $this->components[$component->getTag()][] = $component;
        }
    }
    public function getComponentByTagName($tag){
        if(isset($this->components[$tag])){
            if(count($this->components[$tag]) == 1){
                return $this->components[$tag][0];
            }
            $collection = new elementCollection();
            $collection->addElements($this->components[$tag]);
            return $collection;
        }
    }

}

